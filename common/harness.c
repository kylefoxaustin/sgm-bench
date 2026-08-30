/* harness.c — one binary per implementation, same behaviour for all.
 *
 *   sgm_<impl> LEFT.pgm RIGHT.pgm [options]
 *     -o OUT.pgm        write disparity map
 *     -g GOLDEN.pgm     compare against golden map (exit 2 on mismatch)
 *     -t THREADS        thread count passed to the implementation (0 = default)
 *     -w N              warm-up frames (default 10)
 *     -n N              timed frames   (default 100)
 *     -j RESULTS.json   append a results record
 *     --board NAME      free text recorded in the JSON
 *     --no-roofline     disable the cost gate (it is ON by default)
 *     --roofline-max X  per-phase cost allowed vs calibration (default 1.15)
 *     --roofline-cal F  calibration file (default data/golden/roofline.cal)
 *     --roofline-calibrate   record THIS run as the calibration and exit ok
 *
 * Exit: 0 ok, 1 usage/IO, 2 GOLDEN MISMATCH (wrong answer),
 *       3 ROOFLINE FAIL (right answer, one phase far off its op count).
 *
 * Prints a one-line summary and (optionally) a JSON record containing:
 * impl, board, W, H, D, paths, census, P1, P2, threads, cpu mask, warm/timed
 * counts, median/p95/min ms, fps, Mpix/s, per-stage ms (last frame), output
 * FNV-1a hash, golden match, compiler + flags, git sha (if provided via
 * -DGIT_SHA), timestamp.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sched.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "sgm.h"
#include "roofline.h"

#ifndef CFLAGS_STR
#define CFLAGS_STR "unknown"
#endif
#ifndef GIT_SHA
#define GIT_SHA "unknown"
#endif

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Which CPUs did the work actually run on?
 *
 * sched_getaffinity() on the primary thread cannot answer that. libgomp binds
 * the primary thread in its CONSTRUCTOR, before main() is entered, so under
 * OMP_PROC_BIND -- which scripts/pin.sh sets, and which every published
 * measurement therefore used -- the primary thread's mask is a single core no
 * matter when it is read. Until 2026-08-28 this recorded cpu_mask "0" for every
 * pinned six-core run: true of the thread that asked, false as a record of the
 * run, and wrong in the direction that makes a fast number look like it came
 * from fewer cores.
 *
 * Moving the call to the top of main does NOT fix it -- there is no "earlier"
 * available to us. Measured on an A55: the mask is already count=1 on the first
 * line of main. So take the UNION over the OpenMP team from inside a parallel
 * region, which is the set the work was allowed on and the thing this field is
 * supposed to name. */
static void cpu_mask_string(char *buf, size_t n) {
    cpu_set_t u; CPU_ZERO(&u);
#ifdef _OPENMP
#pragma omp parallel
    {
        cpu_set_t mine; CPU_ZERO(&mine);
        if (sched_getaffinity(0, sizeof mine, &mine) == 0) {
#pragma omp critical
            CPU_OR(&u, &u, &mine);
        }
    }
#else
    if (sched_getaffinity(0, sizeof u, &u) != 0) { snprintf(buf, n, "?"); return; }
#endif
    buf[0] = 0;
    size_t len = 0;
    for (int i = 0; i < CPU_SETSIZE && len + 8 < n; i++)
        if (CPU_ISSET(i, &u)) len += snprintf(buf + len, n - len, "%s%d", len ? "," : "", i);
    if (!len) snprintf(buf, n, "?");
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s LEFT.pgm RIGHT.pgm [-o out.pgm] [-g golden.pgm] [-t threads] [-w warm] [-n timed] [-j results.json] [--board name]\n", argv[0]); return 1; }
    const char *lpath = argv[1], *rpath = argv[2], *opath = NULL, *gpath = NULL, *jpath = NULL, *board = "unknown";
    int threads = 0, warm = 10, timed = 100, roofline = 1, rl_cal_write = 0;
    double max_spread = 1.25;
    double rl_max = SGM_ROOFLINE_DEFAULT_MAX;
    const char *rl_cal = "data/golden/roofline.cal";
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "-o") && i + 1 < argc) opath = argv[++i];
        else if (!strcmp(argv[i], "-g") && i + 1 < argc) gpath = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) warm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) timed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-j") && i + 1 < argc) jpath = argv[++i];
        else if (!strcmp(argv[i], "--board") && i + 1 < argc) board = argv[++i];
        else if (!strcmp(argv[i], "--no-roofline")) roofline = 0;
        else if (!strcmp(argv[i], "--roofline-max") && i + 1 < argc) rl_max = atof(argv[++i]);
        else if (!strcmp(argv[i], "--roofline-cal") && i + 1 < argc) rl_cal = argv[++i];
        else if (!strcmp(argv[i], "--roofline-calibrate")) rl_cal_write = 1;
        else if (!strcmp(argv[i], "--max-spread") && i + 1 < argc) max_spread = atof(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }

    /* -t is authoritative, and what gets REPORTED is what the runtime actually
     * used -- not what was asked for.
     *
     * It was neither, until 2026-08-28. Every impl takes `threads` and drops it
     * ((void)threads), because thread count comes from OpenMP; so -t was parsed,
     * printed in the table, written to the JSON as "threads":N -- and ignored.
     * A `-t 1` run emitted a row LABELLED single-thread holding an all-core
     * timing. Found by the qualcomm session on a Cortex-A78C, where -t 1 / -t 6
     * / -t 8 all returned ~115 ms and would have published a single-thread
     * baseline 2.6x too fast. Note the direction: the error flatters whatever
     * the run is being compared against. Our own numbers escaped only because
     * scripts/pin.sh sets OMP_NUM_THREADS -- a wrapper, not a design.
     *
     * Hence two changes, not one: the flag now binds, AND the count is read
     * back from the runtime, so a future divergence surfaces as a warning
     * instead of a plausible number. */
    int threads_actual = 0;
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    threads_actual = omp_get_max_threads();
    if (threads > 0 && threads_actual != threads)
        fprintf(stderr, "WARNING: requested -t %d but the OpenMP runtime reports %d; "
                        "reporting %d\n", threads, threads_actual, threads_actual);
#else
    if (threads > 0)
        fprintf(stderr, "WARNING: -t %d ignored, built without OpenMP\n", threads);
#endif

    /* after the team size is settled, so the union covers the real team */
    char mask[512]; cpu_mask_string(mask, sizeof mask);

    int W, H, W2, H2;
    uint8_t *L = pgm_read(lpath, &W, &H);
    uint8_t *R = pgm_read(rpath, &W2, &H2);
    if (!L || !R) return 1;
    if (W != W2 || H != H2) { fprintf(stderr, "left/right size mismatch\n"); return 1; }
    uint8_t *disp = calloc((size_t)W * H, 1);
    if (!disp) return 1;

    sgm_stage_times st = { -1, -1, -1, -1, -1, 0 };
    for (int i = 0; i < warm; i++)
        if (SGM_IMPL.run(L, R, W, H, disp, threads, &st)) { fprintf(stderr, "impl failed\n"); return 1; }

    double *ms = malloc(sizeof(double) * (timed > 0 ? timed : 1));
    for (int i = 0; i < timed; i++) {
        double t0 = now_ms();
        if (SGM_IMPL.run(L, R, W, H, disp, threads, &st)) { fprintf(stderr, "impl failed\n"); return 1; }
        ms[i] = now_ms() - t0;
    }
    if (timed == 0) { double t0 = now_ms(); SGM_IMPL.run(L, R, W, H, disp, threads, &st); ms[0] = now_ms() - t0; timed = 1; }

    qsort(ms, timed, sizeof(double), cmp_double);
    double med = ms[timed / 2], mn = ms[0];
    /* ceil(0.95 n) - 1: the 95th-percentile RANK on 0-based sorted data. The
     * naive (int)(0.95 n) is one rank high -- at n <= 20 it returns the MAXIMUM
     * while printing "p95". */
    int p95i = (int)((timed * 95 + 99) / 100) - 1;
    if (p95i < 0) p95i = 0;
    if (p95i >= timed) p95i = timed - 1;
    double p95 = ms[p95i];
    /* DISPERSION. A median with no spread beside it is how an 80%-wrong cell
     * survived publication here. Note `sd` is RMS deviation about the MEDIAN
     * (not the mean), chosen to pair with the median headline; it is labelled
     * rmsd in the JSON accordingly. */
    double sd = 0; for (int i = 0; i < timed; i++) { double d = ms[i] - med; sd += d * d; }
    sd = timed > 1 ? sqrt(sd / (timed - 1)) : 0;
    double spread_hi = med > 0 ? p95 / med : 0, spread_lo = med > 0 ? mn / med : 0;
    int unstable = (spread_hi > max_spread);
    double fps = 1000.0 / med, mpix = (double)W * H / 1e6 * fps;

    uint64_t hash = fnv1a64(disp, (size_t)W * H);
    int golden_match = -1;
    if (gpath) {
        int gw, gh; uint8_t *G = pgm_read(gpath, &gw, &gh);
        if (!G) return 1;
        if (gw != W || gh != H) golden_match = 0;
        else {
            golden_match = 1;
            size_t diff = 0, first = (size_t)-1;
            for (size_t i = 0; i < (size_t)W * H; i++) if (G[i] != disp[i]) { diff++; if (first == (size_t)-1) first = i; }
            if (diff) { golden_match = 0; fprintf(stderr, "GOLDEN MISMATCH: %zu pixels differ, first at (%zu,%zu) got %d want %d\n",
                                               diff, first % W, first / W, disp[first], G[first]); }
        }
        free(G);
    }
    if (opath && pgm_write(opath, disp, W, H)) return 1;

    printf("%-12s %dx%d D=%d paths=%d th=%d  median %8.2f ms  p95 %8.2f  min %8.2f  sd %6.3f  spread %.2f/%.2f%s  fps %7.2f  Mpix/s %7.2f  hash %016llx%s\n",
           SGM_IMPL.name, W, H, SGM_D, SGM_PATHS,
           st.threads_used > 0 ? st.threads_used : threads_actual,
           med, p95, mn, sd, spread_hi, spread_lo, unstable ? " UNSTABLE" : "", fps, mpix,
           (unsigned long long)hash, golden_match == 1 ? "  GOLDEN OK" : golden_match == 0 ? "  GOLDEN FAIL" : "");

    /* The COST gate. Runs by default and in the same invocation as the timing,
     * for the same reason the hash check does: a gate nobody remembers to run
     * is one of the ways a check reads green while being worthless. */
    int rl_fail = 0, rl_armed = 0;
    if (rl_cal_write)
        sgm_roofline_calibrate(rl_cal, SGM_IMPL.name, board, W, H, &st);
    else if (roofline)
        rl_fail = sgm_roofline_check(rl_cal, SGM_IMPL.name, board, W, H, &st,
                                     rl_max, 1, &rl_armed);

    if (jpath) {
        FILE *f = fopen(jpath, "a");
        if (!f) { fprintf(stderr, "cannot open %s\n", jpath); return 1; }
        time_t now = time(NULL); char ts[64]; strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
        fprintf(f, "{\"impl\":\"%s\",\"board\":\"%s\",\"W\":%d,\"H\":%d,\"D\":%d,\"paths\":%d,\"census\":\"%dx%d\",\"P1\":%d,\"P2\":%d,"
                   "\"threads\":%d,\"threads_requested\":%d,\"cpu_mask\":\"%s\",\"warm\":%d,\"timed\":%d,"
                   "\"median_ms\":%.3f,\"p95_ms\":%.3f,\"min_ms\":%.3f,\"rmsd_ms\":%.4f,"
                   "\"spread_hi\":%.3f,\"spread_lo\":%.3f,\"unstable\":%d,"
                   "\"transfer_ms\":%.3f,\"threads_used\":%d,\"fps\":%.3f,\"mpix_s\":%.3f,"
                   "\"census_ms\":%.3f,\"cost_ms\":%.3f,\"aggregate_ms\":%.3f,\"argmin_ms\":%.3f,"
                   "\"hash\":\"%016llx\",\"golden_match\":%d,\"roofline_ok\":%d,\"cflags\":\"%s\",\"git\":\"%s\",\"ts\":\"%s\"}\n",
                SGM_IMPL.name, board, W, H, SGM_D, SGM_PATHS, SGM_CENSUS_W, SGM_CENSUS_H, SGM_P1, SGM_P2,
                st.threads_used > 0 ? st.threads_used : threads_actual, threads,
                mask, warm, timed, med, p95, mn, sd,
                spread_hi, spread_lo, unstable, st.transfer_ms, st.threads_used, fps, mpix,
                st.census_ms, st.cost_ms, st.aggregate_ms, st.argmin_ms,
                (unsigned long long)hash, golden_match, rl_armed ? !rl_fail : -1, CFLAGS_STR, GIT_SHA, ts);
        fclose(f);
    }
    free(L); free(R); free(disp); free(ms);
    /* 2 = wrong answer, 3 = right answer produced too slowly. Distinct codes
     * because they are distinct failures and a caller should be able to tell
     * them apart; correctness outranks cost when both fire. */
    if (golden_match == 0) return 2;
    return rl_fail ? 3 : 0;
}
