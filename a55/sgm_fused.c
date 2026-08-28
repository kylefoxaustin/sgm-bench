/* sgm_fused.c — the structural rewrite: fuse the path pairs, kill the U plane.
 *
 * ---------------------------------------------------------------------------
 * WHAT CHANGES AND WHY
 * ---------------------------------------------------------------------------
 * The previous version stores TWO 133 MB planes (D and U) purely to buy
 * parallelism, because U and D are vertical recurrences and the only free axis
 * for them is columns. That costs 532 MB of write+read per frame and evaluates
 * the Hamming cost FOUR times per pixel (once per sweep).
 *
 * Two observations collapse that:
 *
 *   1. L and U are BOTH forward sweeps (left->right, top->bottom); R and D are
 *      both backward. So one forward traversal can produce L AND U from a
 *      single cost evaluation, and one backward traversal produces R AND D.
 *      Cost evaluations: 4x -> 2x per pixel.
 *
 *   2. A vertical recurrence does not actually need a materialised plane to be
 *      parallel. Split the rows into T horizontal BANDS and software-pipeline
 *      them: thread t may start band t as soon as thread t-1 has finished band
 *      t-1's first row, because all U needs from above is ONE row of state
 *      (W * D bytes). The pipeline fill costs T rows out of H -- 8 of 1080,
 *      about 0.7%.
 *
 * So the forward pass keeps only a per-band carry row, and the U plane
 * disappears entirely. One 133 MB plane remains (the backward pass's, which
 * genuinely cannot be consumed on the fly because it is produced bottom-up and
 * consumed top-down).
 *
 * ---------------------------------------------------------------------------
 * PREDICTION, RECORDED BEFORE MEASUREMENT
 * ---------------------------------------------------------------------------
 * 10-25% on A55, possibly more on A720. Deliberately conservative: the
 * stream-count argument that originally motivated this was FALSIFIED by the
 * interleaved-plane experiment (13% worse, L3 refills +17%), so the only
 * surviving argument is the traffic/work one. Every magnitude estimate I made
 * today was too optimistic, so this one is anchored low on purpose.
 * KILL CRITERION: below +5% on both cores, revert.
 */
#include "sgm.h"
#include "census_neon.h"
#include "hamming_neon.h"
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

#define DREGS (SGM_D / 16)
#ifndef SGM_BW
#define SGM_BW 192
#endif
#ifndef SGM_NT
#define SGM_NT 1
#endif

static void *amalloc(size_t n)
{ void *p = NULL; return posix_memalign(&p, 64, n) ? NULL : p; }

static inline void store_plane(uint8_t *p, const uint8x16_t *v)
{
#if SGM_NT && defined(__aarch64__)
    for (int r = 0; r < DREGS; r += 2)
        __asm__ volatile("stnp %q0, %q1, [%2]"
                         :: "w"(v[r]), "w"(v[r + 1]), "r"(p + r * 16) : "memory");
#else
    for (int r = 0; r < DREGS; r++) vst1q_u8(p + r * 16, v[r]);
#endif
}

static inline void step_neon(const uint8x16_t *prev, const uint8x16_t *cost,
                             uint8x16_t *cur, uint8x16_t mn)
{
    const uint8x16_t p1 = vdupq_n_u8(SGM_P1), big = vdupq_n_u8(255);
    const uint8x16_t e  = vqaddq_u8(mn, vdupq_n_u8(SGM_P2));
    for (int r = 0; r < DREGS; r++) {
        uint8x16_t lo = (r == 0) ? vextq_u8(big, prev[0], 15)
                                 : vextq_u8(prev[r - 1], prev[r], 15);
        uint8x16_t hi = (r == DREGS - 1) ? vextq_u8(prev[r], big, 1)
                                         : vextq_u8(prev[r], prev[r + 1], 1);
        uint8x16_t m = prev[r];
        m = vminq_u8(m, vqaddq_u8(lo, p1));
        m = vminq_u8(m, vqaddq_u8(hi, p1));
        m = vminq_u8(m, e);
        cur[r] = vqaddq_u8(cost[r], vsubq_u8(m, mn));
    }
}

static inline uint8x16_t hmin_bcast(const uint8x16_t *v)
{
    uint8x16_t m = v[0];
    for (int r = 1; r < DREGS; r++) m = vminq_u8(m, v[r]);
    return vdupq_n_u8(vminvq_u8(m));
}

static inline void load_cost(const uint64_t *cl, const uint64_t *cr,
                             int x, uint8x16_t *out)
{
    if (x >= SGM_D - 1 + 15) {
        for (int r = 0; r < DREGS; r++) out[r] = ham16_neon(cl, cr, x, r * 16);
    } else {
        uint8_t tmp[SGM_D];
        for (int d = 0; d < SGM_D; d++) tmp[d] = ham_scalar(cl, cr, x, d);
        for (int r = 0; r < DREGS; r++) out[r] = vld1q_u8(tmp + r * 16);
    }
}

static int fused_run(const uint8_t *left, const uint8_t *right, int W, int H,
                     uint8_t *disp, int threads, sgm_stage_times *t)
{
    (void)threads;
    const size_t N = (size_t)W * H;
    uint64_t *cl = amalloc(N * 8), *cr = amalloc(N * 8);
    uint8_t  *Bplane = amalloc(N * SGM_D);      /* backward pass: R+D summed */
    if (!cl || !cr || !Bplane) return -1;

    double t0 = now_ms();
    census_neon(left,  cl, W, H);
    census_neon(right, cr, W, H);
    double t1 = now_ms();

    /* ---- BACKWARD pass: R (right->left, per row) and D (bottom->top).
     * D is the vertical recurrence that must be materialised; R is fused into
     * the same traversal so the cost is evaluated once. We store the SUM of the
     * two, as uint8 is not enough (2 x 254) -- so this plane is uint8 per path
     * and we keep only D, adding R on the fly in the forward pass would need
     * R again. Instead: store D only, and recompute R in the forward pass from
     * the same cost we already have. That keeps the plane at 133 MB and still
     * evaluates cost twice per pixel overall. */
    {
        const int BW = SGM_BW;
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            uint8_t *st = amalloc((size_t)BW * SGM_D);
            uint8x16_t cur[DREGS], c[DREGS], pv[DREGS];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int xb = 0; xb < W; xb += BW) {
                const int xe = (xb + BW < W) ? xb + BW : W;
                for (int y = H - 1; y >= 0; y--) {
                    const uint64_t *rl = cl + (size_t)y * W, *rr = cr + (size_t)y * W;
                    for (int x = xb; x < xe; x++) {
                        uint8_t *sp = st + (size_t)(x - xb) * SGM_D;
                        load_cost(rl, rr, x, c);
                        if (y == H - 1) for (int r = 0; r < DREGS; r++) cur[r] = c[r];
                        else {
                            for (int r = 0; r < DREGS; r++) pv[r] = vld1q_u8(sp + r * 16);
                            step_neon(pv, c, cur, hmin_bcast(pv));
                        }
                        for (int r = 0; r < DREGS; r++) vst1q_u8(sp + r * 16, cur[r]);
                        store_plane(Bplane + ((size_t)y * W + x) * SGM_D, cur);
                    }
                }
            }
            free(st);
        }
    }

    /* ---- FORWARD pass: L, R, U on the fly + stored D, argmin.
     *
     * U IS MADE PARALLEL BY A PER-ROW WAVEFRONT, NOT BY A PLANE.
     *
     * (An earlier draft used contiguous row BANDS. That is wrong: band b+1
     * needs the U state of band b's LAST row, so it cannot start until band b
     * finishes -- serialisation, not pipelining. Caught before measuring.)
     *
     * Rows are handed out round-robin (schedule(static,1)), so at steady state
     * T threads occupy T consecutive rows. The thread on row y spins until row
     * y-1 has published its U, then reads it from a rotating slot. The lag is
     * ONE row and the work per row is W pixels, so the pipe fills immediately
     * and stays full. L and R are row-local and need no handshake at all.
     * State crossing rows: one W*D row, held in (T+1) rotating buffers
     * (~861 KB at T=6) instead of a 133 MB plane. */
    {
        int T = 1;
#ifdef _OPENMP
#pragma omp parallel
        { 
#pragma omp master
            T = omp_get_num_threads();
        }
#endif
        if (T < 1) T = 1;
        const int NSLOT = T + 1;
        uint8_t **slot = calloc(NSLOT, sizeof(*slot));
        for (int i = 0; i < NSLOT; i++) slot[i] = amalloc((size_t)W * SGM_D);
        volatile int ready = -1;          /* highest row whose U is published */

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            uint8_t *rowR = amalloc((size_t)W * SGM_D);
            uint8x16_t c[DREGS], pv[DREGS], cur[DREGS], pvL[DREGS], curL[DREGS], curU[DREGS];
#ifdef _OPENMP
#pragma omp for schedule(static, 1)
#endif
            for (int y = 0; y < H; y++) {
                if (y > 0)
                    while (__atomic_load_n(&ready, __ATOMIC_ACQUIRE) < y - 1)
                        __asm__ volatile("yield" ::: "memory");

                const uint64_t *rl = cl + (size_t)y * W, *rr = cr + (size_t)y * W;
                uint8_t *myU = slot[y % NSLOT];
                const uint8_t *upU = slot[(y + NSLOT - 1) % NSLOT];

                for (int x = W - 1; x >= 0; x--) {          /* R: right -> left */
                    load_cost(rl, rr, x, c);
                    if (x == W - 1) for (int r = 0; r < DREGS; r++) cur[r] = c[r];
                    else            step_neon(pv, c, cur, hmin_bcast(pv));
                    for (int r = 0; r < DREGS; r++) {
                        vst1q_u8(rowR + (size_t)x * SGM_D + r * 16, cur[r]);
                        pv[r] = cur[r];
                    }
                }

                for (int x = 0; x < W; x++) {               /* L + U + sum */
                    load_cost(rl, rr, x, c);
                    if (x == 0) for (int r = 0; r < DREGS; r++) curL[r] = c[r];
                    else        step_neon(pvL, c, curL, hmin_bcast(pvL));

                    if (y == 0) for (int r = 0; r < DREGS; r++) curU[r] = c[r];
                    else {
                        uint8x16_t pu[DREGS];
                        for (int r = 0; r < DREGS; r++)
                            pu[r] = vld1q_u8(upU + (size_t)x * SGM_D + r * 16);
                        step_neon(pu, c, curU, hmin_bcast(pu));
                    }

                    const uint8_t *dp = Bplane + ((size_t)y * W + x) * SGM_D;
                    const uint8_t *Rb = rowR + (size_t)x * SGM_D;
                    uint16x8_t sum[DREGS * 2];
                    for (int r = 0; r < DREGS; r++) {
                        uint8x16_t Rv = vld1q_u8(Rb + r * 16);
                        uint8x16_t Dv = vld1q_u8(dp + r * 16);
                        uint16x8_t lo = vaddl_u8(vget_low_u8(curL[r]), vget_low_u8(curU[r]));
                        uint16x8_t hi = vaddl_high_u8(curL[r], curU[r]);
                        lo = vaddq_u16(lo, vaddl_u8(vget_low_u8(Rv), vget_low_u8(Dv)));
                        hi = vaddq_u16(hi, vaddl_high_u8(Rv, Dv));
                        sum[r * 2] = lo; sum[r * 2 + 1] = hi;
                        pvL[r] = curL[r];
                        vst1q_u8(myU + (size_t)x * SGM_D + r * 16, curU[r]);
                    }
                    uint16x8_t mv = sum[0];
                    for (int q = 1; q < DREGS * 2; q++) mv = vminq_u16(mv, sum[q]);
                    uint16_t best = vminvq_u16(mv);
                    uint16x8_t bv = vdupq_n_u16(best);
                    int bestd = 0;
                    for (int q = 0; q < DREGS * 2; q++) {
                        uint8x8_t n = vmovn_u16(vceqq_u16(sum[q], bv));
                        uint64_t m = vget_lane_u64(vreinterpret_u64_u8(n), 0);
                        if (m) { bestd = q * 8 + (__builtin_ctzll(m) >> 3); break; }
                    }
                    disp[(size_t)y * W + x] = (uint8_t)bestd;
                }
                __atomic_store_n(&ready, y, __ATOMIC_RELEASE);
            }
            free(rowR);
        }
        for (int i = 0; i < NSLOT; i++) free(slot[i]);
        free(slot);
    }
    double t2 = now_ms();

    if (t) { t->census_ms = t1 - t0; t->cost_ms = -1;
             t->aggregate_ms = t2 - t1; t->argmin_ms = -1; }
    free(cl); free(cr); free(Bplane);
    return 0;
}

const sgm_impl SGM_IMPL = { "fused", fused_run };

#else
static int fused_run(const uint8_t *l, const uint8_t *r, int W, int H,
                     uint8_t *d, int th, sgm_stage_times *t)
{ (void)l;(void)r;(void)W;(void)H;(void)d;(void)th;(void)t; return -1; }
const sgm_impl SGM_IMPL = { "fused(no-neon)", fused_run };
#endif
