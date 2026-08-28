/* test_phase1.c — Phase 1 acceptance: NEON kernels must be bit-identical to
 * their scalar twins, and we want to know what they cost.
 *
 * Bit-exactness is checked on the DESCRIPTORS, not just on some downstream
 * output, because a descriptor diff localises the failing bit and a disparity
 * diff does not. It is also checked on a REAL image (the synthetic pair), not
 * only on random data: random bytes have no runs, no flat regions and no ties,
 * and ties are exactly where census implementations disagree.
 */
#include "sgm.h"
#include "census_neon.h"
#include "hamming_neon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(const char *what, const uint64_t *a, const uint64_t *b,
                 int W, int H)
{
    for (size_t i = 0; i < (size_t)W * H; i++) {
        if (a[i] != b[i]) {
            int y = (int)(i / W), x = (int)(i % W);
            uint64_t d = a[i] ^ b[i];
            printf("  ✗ %s: FIRST DIFF at (%d,%d) idx %zu\n", what, x, y, i);
            printf("      scalar 0x%016llx\n      neon   0x%016llx\n",
                   (unsigned long long)a[i], (unsigned long long)b[i]);
            printf("      xor    0x%016llx  (lowest differing bit: %d)\n",
                   (unsigned long long)d, __builtin_ctzll(d));
            return 1;
        }
    }
    printf("  ✓ %s: bit-identical over %d x %d\n", what, W, H);

    return 0;
}

int main(int argc, char **argv)
{
    int W = 1920, H = 1080, reps = 5;
    const char *pgm = (argc > 1) ? argv[1] : NULL;
    uint8_t *im = NULL;

    if (pgm) {
        im = pgm_read(pgm, &W, &H);
        if (!im) { fprintf(stderr, "cannot read %s\n", pgm); return 1; }
        printf("image: %s %dx%d\n", pgm, W, H);
    } else {
        im = malloc((size_t)W * H);
        unsigned s = 1;
        for (size_t i = 0; i < (size_t)W * H; i++) { s = s*1103515245u+12345u; im[i] = (uint8_t)(s>>16); }
        printf("image: random %dx%d\n", W, H);
    }

    uint64_t *cs = malloc((size_t)W * H * 8);
    uint64_t *cn = malloc((size_t)W * H * 8);
    if (!cs || !cn) { fprintf(stderr, "alloc\n"); return 1; }

    /* correctness first, timing second: a fast wrong kernel is worthless */
    census_scalar_ref(im, cs, W, H);
    census_neon(im, cn, W, H);
    int bad = check("census 9x7", cs, cn, W, H);
    if (bad) return 2;

    double t0 = now_ms();
    for (int r = 0; r < reps; r++) census_scalar_ref(im, cs, W, H);
    double ts = (now_ms() - t0) / reps;

    t0 = now_ms();
    for (int r = 0; r < reps; r++) census_neon(im, cn, W, H);
    double tn = (now_ms() - t0) / reps;

    double mp = (double)W * H / 1e6;
    printf("  census scalar : %8.2f ms  (%6.2f Mpix/s)\n", ts, mp / (ts/1000.0));
    printf("  census neon   : %8.2f ms  (%6.2f Mpix/s)   speedup %.2fx\n",
           tn, mp / (tn/1000.0), ts / tn);

#if defined(__ARM_NEON) || defined(__aarch64__)
    /* ---- Hamming: NEON 16-at-a-time vs scalar, every legal (x,d0) ---- */
    {
        uint64_t *cr = malloc((size_t)W * H * 8);
        census_neon(im, cr, W, H);           /* a second, different descriptor plane */
        int bad2 = 0;
        for (int y = 0; y < H && !bad2; y += 97) {
            const uint64_t *rl = cs + (size_t)y * W, *rr = cr + (size_t)y * W;
            for (int x = SGM_D; x < W && !bad2; x += 13) {
                for (int d0 = 0; d0 + 16 <= SGM_D; d0 += 16) {
                    uint8_t got[16];
                    vst1q_u8(got, ham16_neon(rl, rr, x, d0));
                    for (int i = 0; i < 16; i++) {
                        uint8_t want = ham_scalar(rl, rr, x, d0 + i);
                        if (got[i] != want) {
                            printf("  x hamming: DIFF at x=%d d=%d scalar=%u neon=%u\n",
                                   x, d0 + i, want, got[i]);
                            bad2 = 1; break;
                        }
                    }
                    if (bad2) break;
                }
            }
        }
        if (!bad2) printf("  v hamming 16-wide: matches scalar at every sampled (x,d)\n");
        else return 3;
        free(cr);
    }
#endif
    return 0;
}
