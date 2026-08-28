/* sgm_colmajor.c — SGM with the vector axis on COLUMNS instead of DISPARITY.
 *
 * WHY THIS LAYOUT EXISTS. 95emulator's NEON implementation puts d in the lanes
 * (16 disparities per register, DREGS=D/16 registers per pixel). Two costs
 * follow from that choice, both of which they identified:
 *   (a) min over d is a CROSS-LANE reduce (vminvq) — the one operation that does
 *       not get cheaper as the vector gets wider.
 *   (d) the d-1 / d+1 neighbours cross register boundaries, needing vextq and
 *       a 255 sentinel shifted in at each end.
 *
 * Putting x in the lanes and iterating d as a scalar loop removes BOTH:
 *   · min over d becomes D sequential vminq over whole vectors — no reduce.
 *   · d-1 / d+1 become ARRAY INDICES into the per-d state — no lane crossing.
 * The price is state: L[d][lane] must be resident, D*VLEN bytes per block,
 * versus DREGS registers per pixel.
 *
 * This is written as the ARM/NEON twin of the HVX kernel it is designed for.
 * On HVX the lane count is 128 rather than 16, which is where the layout pays:
 * the same D=64 scalar loop covers 8x more columns per pass, and VTCM holds the
 * 8 KB of state comfortably. Validating it here, bit-exact against the shared
 * golden, is what makes the HVX port a mechanical translation rather than a
 * rewrite with an unproven algorithm inside it.
 *
 * Bit-exactness requirement: identical arithmetic to the reference —
 * saturating u8 adds, the same P1/P2, the same tie-goes-to-lowest-d argmin.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include "sgm.h"
#include "hamming_neon.h"

#define VL 16                     /* NEON lanes; HVX twin uses 128 */
#ifndef CM_BW
#define CM_BW 128                 /* columns per block: multiple of VL */
#endif

void census_neon(const uint8_t *im, uint64_t *out, int W, int H);

static void *amalloc(size_t n){ void *p=NULL; if(posix_memalign(&p,128,n)) return NULL; return p; }

/* One SGM step for a block of BW columns at a fixed y, for ONE path direction.
 * L    : [D][BW] previous-pixel aggregated cost (u8)
 * C    : [D][BW] matching cost at this pixel   (u8)
 * out  : [D][BW] new aggregated cost
 * mn   : [BW]    min over d of L, per column
 * Everything is vector over columns; d is the scalar loop index. */
static inline void step_cols(const uint8_t *L, const uint8_t *C, uint8_t *out,
                             const uint8_t *mn, int BW)
{
    const uint8x16_t p1 = vdupq_n_u8(SGM_P1);
    const uint8x16_t p2 = vdupq_n_u8(SGM_P2);
    const uint8x16_t big = vdupq_n_u8(255);
    for (int d = 0; d < SGM_D; d++) {
        const uint8_t *Ld = L + (size_t)d * BW;
        const uint8_t *Lm = (d == 0)         ? NULL : Ld - BW;   /* d-1 */
        const uint8_t *Lp = (d == SGM_D - 1) ? NULL : Ld + BW;   /* d+1 */
        for (int x = 0; x < BW; x += VL) {
            uint8x16_t cur = vld1q_u8(Ld + x);
            uint8x16_t lo  = Lm ? vld1q_u8(Lm + x) : big;        /* array index, not vext */
            uint8x16_t hi  = Lp ? vld1q_u8(Lp + x) : big;
            uint8x16_t m   = vdupq_n_u8(0);
            uint8x16_t mnv = vld1q_u8(mn + x);
            m = cur;
            m = vminq_u8(m, vqaddq_u8(lo, p1));
            m = vminq_u8(m, vqaddq_u8(hi, p1));
            m = vminq_u8(m, vqaddq_u8(mnv, p2));
            vst1q_u8(out + (size_t)d * BW + x,
                     vqaddq_u8(vld1q_u8(C + (size_t)d * BW + x), vsubq_u8(m, mnv)));
        }
    }
}

/* min over d, per column — D vector mins, NO cross-lane reduce. */
static inline void colmin(const uint8_t *L, uint8_t *mn, int BW)
{
    for (int x = 0; x < BW; x += VL) vst1q_u8(mn + x, vld1q_u8(L + x));
    for (int d = 1; d < SGM_D; d++) {
        const uint8_t *Ld = L + (size_t)d * BW;
        for (int x = 0; x < BW; x += VL)
            vst1q_u8(mn + x, vminq_u8(vld1q_u8(mn + x), vld1q_u8(Ld + x)));
    }
}

/* Matching cost for a block of columns at row y, laid out [D][BW]. */
static inline void cost_cols(const uint64_t *cl, const uint64_t *cr,
                             int W, int y, int xb, int BW, uint8_t *C)
{
    const uint64_t *rowl = cl + (size_t)y * W, *rowr = cr + (size_t)y * W;
    for (int d = 0; d < SGM_D; d++) {
        uint8_t *Cd = C + (size_t)d * BW;
        for (int i = 0; i < BW; i++) {
            int x = xb + i;
            Cd[i] = (x >= W || x - d < 0) ? (uint8_t)SGM_COST_INVALID
                   : (uint8_t)__builtin_popcountll(rowl[x] ^ rowr[x - d]);
        }
    }
}

/* Accumulate one path's contribution into the running sum S[y][x][d].
 * S is u16 because four paths of u8 can exceed 255. */
static inline void accum(uint16_t *S, const uint8_t *L, int BW, int W,
                         int xb, int y)
{
    for (int d = 0; d < SGM_D; d++) {
        const uint8_t *Ld = L + (size_t)d * BW;
        for (int i = 0; i < BW; i++) {
            int x = xb + i;
            if (x < W) S[((size_t)y * W + x) * SGM_D + d] += Ld[i];
        }
    }
}

static int colmajor_run(const uint8_t *left, const uint8_t *right, int W, int H,
                        uint8_t *disp, int threads, sgm_stage_times *t)
{
    (void)threads;
    const size_t N = (size_t)W * H;
    uint64_t *cl = amalloc(N * 8), *cr = amalloc(N * 8);
    uint16_t *S  = amalloc(N * SGM_D * sizeof(uint16_t));
    if (!cl || !cr || !S) return -1;
    memset(S, 0, N * SGM_D * sizeof(uint16_t));

    double t0 = now_ms();
    census_neon(left,  cl, W, H);
    census_neon(right, cr, W, H);
    double t1 = now_ms();

    const int BW = CM_BW;
    const int NB = (W + BW - 1) / BW;
    double tv0 = now_ms();

    /* ---- VERTICAL paths (D: bottom->top, U: top->bottom).
     * Independent across columns, so a column BLOCK is the natural parallel
     * unit AND the natural vector unit — the same axis serves both, which is
     * the point of this layout. ---- */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < NB; b++) {
        const int xb = b * BW;
        uint8_t *L  = amalloc((size_t)SGM_D * BW);
        uint8_t *Ln = amalloc((size_t)SGM_D * BW);
        uint8_t *C  = amalloc((size_t)SGM_D * BW);
        uint8_t *mn = amalloc(BW);
        if (!L || !Ln || !C || !mn) continue;
        for (int dir = 0; dir < 2; dir++) {
            const int y0 = dir ? 0 : H - 1, y1 = dir ? H : -1, dy = dir ? 1 : -1;
            for (int y = y0; y != y1; y += dy) {
                cost_cols(cl, cr, W, y, xb, BW, C);
                if (y == y0) memcpy(L, C, (size_t)SGM_D * BW);
                else { colmin(L, mn, BW); step_cols(L, C, Ln, mn, BW);
                       memcpy(L, Ln, (size_t)SGM_D * BW); }
                accum(S, L, BW, W, xb, y);
            }
        }
        free(L); free(Ln); free(C); free(mn);
    }

    double tv1 = now_ms();   /* end of VERTICAL (vectorised, column-major) */

    /* ---- HORIZONTAL paths (L: left->right, R: right->left).
     * Independent across ROWS, so rows are the parallel unit. The vector axis
     * is still columns, but a horizontal path is serial in x — so here the
     * block is one column wide and we fall back to the reference's d-in-lanes
     * shape. This asymmetry is inherent to SGM, not to the layout. ---- */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < H; y++) {
        uint8_t L[SGM_D], Ln[SGM_D], C[SGM_D];
        for (int dir = 0; dir < 2; dir++) {
            const int x0 = dir ? 0 : W - 1, x1 = dir ? W : -1, dx = dir ? 1 : -1;
            for (int x = x0; x != x1; x += dx) {
                for (int d = 0; d < SGM_D; d++)
                    C[d] = (x - d < 0) ? (uint8_t)SGM_COST_INVALID
                         : (uint8_t)__builtin_popcountll(cl[(size_t)y*W+x] ^ cr[(size_t)y*W+x-d]);
                if (x == x0) memcpy(L, C, SGM_D);
                else {
                    uint8_t mn = 255;
                    for (int d = 0; d < SGM_D; d++) if (L[d] < mn) mn = L[d];
                    for (int d = 0; d < SGM_D; d++) {
                        unsigned lo = (d == 0)         ? 255 : L[d-1];
                        unsigned hi = (d == SGM_D - 1) ? 255 : L[d+1];
                        unsigned a = L[d];
                        unsigned b = lo + SGM_P1; if (b > 255) b = 255;
                        unsigned c = hi + SGM_P1; if (c > 255) c = 255;
                        unsigned e = mn  + SGM_P2; if (e > 255) e = 255;
                        unsigned m = a; if (b < m) m = b; if (c < m) m = c; if (e < m) m = e;
                        unsigned v = C[d] + (m - mn); if (v > 255) v = 255;
                        Ln[d] = (uint8_t)v;
                    }
                    memcpy(L, Ln, SGM_D);
                }
                for (int d = 0; d < SGM_D; d++) S[((size_t)y*W+x)*SGM_D+d] += L[d];
            }
        }
    }
    double t2 = now_ms();
    fprintf(stderr, "[colmajor] VERTICAL(vec) %.2f ms | HORIZONTAL(scalar) %.2f ms\n", tv1-tv0, t2-tv1);

    /* argmin — ties go to the LOWEST d, matching the oracle. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < N; i++) {
        const uint16_t *s = S + i * SGM_D;
        uint16_t best = s[0]; int bd = 0;
        for (int d = 1; d < SGM_D; d++) if (s[d] < best) { best = s[d]; bd = d; }
        disp[i] = (uint8_t)bd;
    }
    double t3 = now_ms();

    if (t) { t->census_ms = t1-t0; t->cost_ms = -1; t->aggregate_ms = t2-t1; t->argmin_ms = t3-t2; }
    free(cl); free(cr); free(S);
    return 0;
}

const sgm_impl SGM_IMPL = { "colmajor", colmajor_run };
