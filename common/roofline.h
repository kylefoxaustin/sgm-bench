/* roofline.h — the SECOND gate: a predicted cost, checked in the same run.
 *
 * A golden hash constrains OUTPUTS. It cannot constrain how they were produced,
 * so it is structurally blind to a kernel that is correct and slow. Two
 * half-scalar kernels passed the hash gate during this campaign; one handed 6.7%
 * of its pixels to a scalar fallback that ate ~50% of the phase, output perfect.
 * Correctness gates catch wrong answers; only a predicted cost catches
 * slow-but-right.
 *
 * THE MODEL. Each phase has a scalar-equivalent op count fixed by the workload:
 *
 *     census     2 images x W x H x CENSUS_BITS  compare-and-pack
 *     cost       W x H x D                       Hamming popcounts
 *     aggregate  PATHS x W x H x D x 4           min / add / sub / renormalise
 *     argmin     W x H x D                       compares
 *
 * Measured time / op count = ns per scalar-equivalent op: an efficiency with a
 * FIXED denominator. Shares cannot do this job -- a share is a ratio against a
 * total that also moves, so a phase can read "6.9% of runtime", look finished,
 * and hide a 5.8x.
 *
 * ⚠️ WHY THIS IS NOT A SPREAD CHECK, which is what it was first built as and
 * which DOES NOT WORK. Comparing phases against each other seems machine-
 * independent and needs no calibration. It is also structurally incapable of
 * catching a regression in the FASTEST phase: slowing the best phase moves it
 * TOWARD the worst and the spread SHRINKS. Measured, by planting exactly the
 * half-scalar census bug described above: healthy spread 1.62x, planted spread
 * 1.26x. The defect made the gate look healthier. A relative gate can only ever
 * catch a phase that becomes the worst one.
 *
 * So each phase is checked against ITS OWN calibrated budget, absolutely.
 *
 * THRESHOLD PICKED BY PLANTING, NOT BY TASTE. On 6x A55, five repeats of a
 * healthy build: census 0.0819-0.0832 ns/op (1.6% spread), aggregate
 * 0.1321-0.1334 (1.0%). Planting the half-scalar census -- the vector loop
 * stopped 124 columns short, handing them to the scalar tail, exactly the
 * observed bug -- measured 0.1041 against a calibrated 0.0835: 1.25x. So 1.15
 * sits ~15x above the run-to-run noise and below the defect. VERIFIED to fire:
 * the planted build exits 3 with GOLDEN OK, which is the whole point. A gate
 * tuned by taste passes everything forever and looks exactly like a clean
 * codebase.
 *
 * FAILS CLOSED, AND SAYS SO WHEN IT CANNOT RUN. Fused phases (reported as -1,
 * as this harness does for cost and argmin) have their ops folded into
 * aggregate, which is the phase that did the work. If no phase reports a time,
 * or no calibration exists for this impl+board, the gate is NOT ARMED and says
 * so loudly on stderr -- an unarmed gate that printed nothing would be one more
 * check that reads green while being worthless.
 *
 * WHAT IT CANNOT DO: it compares against a calibration recorded from a build
 * someone believed was good. It catches REGRESSION from that point, not an
 * implementation that was slow when calibrated.
 */
#ifndef SGM_ROOFLINE_H_INCLUDED
#define SGM_ROOFLINE_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sgm_params.h"

#define SGM_ROOFLINE_DEFAULT_MAX 1.15
#define SGM_RL_PHASES 4

static const char *sgm_rl_names[SGM_RL_PHASES] = { "census", "cost", "aggregate", "argmin" };

static inline void sgm_rl_ops(int W, int H, const sgm_stage_times *t, double *ops)
{
    const double px = (double)W * H;
    ops[0] = 2.0 * px * SGM_CENSUS_BITS;
    ops[1] = px * SGM_D;
    ops[2] = (double)SGM_PATHS * px * SGM_D * 4.0;
    ops[3] = px * SGM_D;
    const double ms[SGM_RL_PHASES] = { t->census_ms, t->cost_ms, t->aggregate_ms, t->argmin_ms };
    for (int i = 0; i < SGM_RL_PHASES; i++)
        if (i != 2 && ms[i] < 0) { ops[2] += ops[i]; ops[i] = 0; }
}

/* Write this run's ns/op as the calibration for impl+board. */
static inline int sgm_roofline_calibrate(const char *path, const char *impl, const char *board,
                                         int W, int H, const sgm_stage_times *t)
{
    double ops[SGM_RL_PHASES]; sgm_rl_ops(W, H, t, ops);
    const double ms[SGM_RL_PHASES] = { t->census_ms, t->cost_ms, t->aggregate_ms, t->argmin_ms };
    FILE *f = fopen(path, "a");
    if (!f) { fprintf(stderr, "cannot write calibration %s\n", path); return 1; }
    for (int i = 0; i < SGM_RL_PHASES; i++) {
        if (ms[i] < 0 || ops[i] == 0) continue;
        fprintf(f, "%s %s %dx%d D%d %s %.6f\n", impl, board, W, H, SGM_D,
                sgm_rl_names[i], ms[i] * 1e6 / ops[i]);
    }
    fclose(f);
    fprintf(stderr, "roofline: calibration written to %s for %s/%s\n", path, impl, board);
    return 0;
}

/* 0 = pass (or not armed), 1 = fail. *armed reports whether it actually ran. */
static inline int sgm_roofline_check(const char *path, const char *impl, const char *board,
                                     int W, int H, const sgm_stage_times *t,
                                     double max_ratio, int verbose, int *armed)
{
    double ops[SGM_RL_PHASES]; sgm_rl_ops(W, H, t, ops);
    const double ms[SGM_RL_PHASES] = { t->census_ms, t->cost_ms, t->aggregate_ms, t->argmin_ms };
    *armed = 0;

    /* Fused phases fold their ops into aggregate -- but if AGGREGATE itself is
     * unreported, those ops have nowhere to go and ~94% of the work would be
     * silently ungated while the gate still printed a green line. Fail closed. */
    if (t->aggregate_ms < 0) {
        fprintf(stderr, "ROOFLINE FAIL: aggregate_ms unreported, so the ops folded "
                        "into it are unaccounted for and most of the work would go "
                        "unchecked. Report a stage time for aggregate.\n");
        return 1;
    }
    int any = 0;
    for (int i = 0; i < SGM_RL_PHASES; i++) if (ms[i] >= 0 && ops[i] > 0) any = 1;
    if (!any) {
        fprintf(stderr, "ROOFLINE NOT ARMED: implementation reported no per-stage times, "
                        "so the cost gate could not run.\n");
        return 0;
    }

    FILE *f = path ? fopen(path, "r") : NULL;
    if (!f) {
        fprintf(stderr, "ROOFLINE NOT ARMED: no calibration file%s%s. "
                        "Run `make roofline-cal` on a build you trust, then this gate arms itself.\n",
                path ? " at " : "", path ? path : "");
        return 0;
    }

    /* The key includes RESOLUTION and D, not just impl+board. ns/op moves ~22%
     * with D and ~1.5x across the resolution sweep, so a calibration taken at
     * one configuration either false-fails another or forces --no-roofline --
     * which is exactly what the sweep script used to do. A cal entry now only
     * matches the configuration it was recorded at. */
    char want[96]; snprintf(want, sizeof want, "%dx%d D%d", W, H, SGM_D);
    double cal[SGM_RL_PHASES]; for (int i = 0; i < SGM_RL_PHASES; i++) cal[i] = -1;
    char li[64], lb[64], lr[64], ld[64], lp[64]; double v;
    while (fscanf(f, "%63s %63s %63s %63s %63s %lf", li, lb, lr, ld, lp, &v) == 6) {
        char got[96]; snprintf(got, sizeof got, "%s %s", lr, ld);
        for (int i = 0; i < SGM_RL_PHASES; i++)
            if (!strcmp(li, impl) && !strcmp(lb, board) && !strcmp(got, want)
                && !strcmp(lp, sgm_rl_names[i])) cal[i] = v;
    }
    fclose(f);

    int have = 0; for (int i = 0; i < SGM_RL_PHASES; i++) if (cal[i] > 0) have = 1;
    if (!have) {
        fprintf(stderr, "ROOFLINE NOT ARMED: no calibration for %s/%s at %dx%d D=%d. "
                        "Run `make roofline-cal` for this configuration.\n",
                impl, board, W, H, SGM_D);
        return 0;
    }

    *armed = 1;
    int fail = 0;
    if (verbose) printf("  roofline:");
    for (int i = 0; i < SGM_RL_PHASES; i++) {
        if (ms[i] < 0 || ops[i] == 0) continue;
        double nspo = ms[i] * 1e6 / ops[i];
        if (cal[i] <= 0) {
            if (verbose) printf("  %s %.4f (uncal)", sgm_rl_names[i], nspo);
            continue;
        }
        double r = nspo / cal[i];
        if (verbose) printf("  %s %.4f/%.4f %.2fx", sgm_rl_names[i], nspo, cal[i], r);
        if (r > max_ratio) {
            fprintf(stderr, "\nROOFLINE FAIL: phase '%s' costs %.4f ns/op against a calibrated "
                            "%.4f (%.2fx, limit %.2fx). It is doing far more work per op than "
                            "this implementation is known to need -- look for a scalar tail, a "
                            "spill, or an unvectorised path. The output may still be bit-exact; "
                            "that is precisely what this gate exists to catch.\n",
                    sgm_rl_names[i], nspo, cal[i], r, max_ratio);
            fail = 1;
        }
    }
    if (verbose) printf("\n");
    return fail;
}

#endif /* SGM_ROOFLINE_H_INCLUDED */
