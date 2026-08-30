/* sgm_a55.c — SGM aggregation tuned for Cortex-A55 (in-order, 128-bit NEON).
 *
 * ---------------------------------------------------------------------------
 * THE STORAGE ARGUMENT, because it determines the whole structure
 * ---------------------------------------------------------------------------
 * U (top->bottom) and D (bottom->top) visit rows in opposite orders, so one of
 * them cannot be consumed on the fly and MUST be materialised. The obvious form
 * stores the uint16 aggregated volume S = W*H*D*2 = 265 MB at 1080p/D=64, and
 * then the benchmark measures LPDDR, not the CPU.
 *
 * So we store the cheapest possible thing: the D-path costs alone, uint8,
 * W*H*D = 133 MB. Everything else is computed on the fly in one top->bottom
 * pass:
 *      L : previous pixel is (x-1,y)  -> lives in registers along the row
 *      R : previous pixel is (x+1,y)  -> a right-to-left sub-pass, one W*D row
 *      U : previous pixel is (x,y-1)  -> one W*D row buffer
 *      D : read back from the stored plane
 * Traffic is then 133 MB written + 133 MB read per frame instead of ~800 MB.
 * That is the floor for 4-path SGM, and quantifying it is half the point of the
 * exercise: it is exactly what a line-buffered hardware block does not pay.
 *
 * ---------------------------------------------------------------------------
 * THE KERNEL
 * ---------------------------------------------------------------------------
 * D=64 disparities = four uint8x16_t, held in registers across the pixel step.
 *      L(d) = C(d) + min( Lp(d), Lp(d-1)+P1, Lp(d+1)+P1, min_k Lp(k)+P2 )
 *                  - min_k Lp(k)
 * d-1 / d+1 come from vextq_u8 across adjacent registers, with 0xFF shifted in
 * at the two ends so the out-of-range neighbours lose every min (this is why
 * the adds must be SATURATING: vqaddq_u8(255,P1) stays 255 instead of wrapping
 * to a small number and winning).
 *
 * Renormalisation keeps L in uint8: the bracket is at most min_k+P2, so
 * L <= COST_MAX + P2 = 62 + 192 = 254. The static assert in sgm_params.h is
 * what guarantees that, and it is not decorative: a P2 of 200 would overflow
 * it (62 + 200 = 262), which is why P2 is 192 here.
 */
#include "sgm.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "census_neon.h"
#include "hamming_neon.h"
#include <stdlib.h>
#include <string.h>

/* 64-byte (cache-line) aligned allocation.
 * MEASURED: plain malloc returns page-aligned+16 on this glibc, so a 64-byte
 * per-pixel plane access starts at offset 16 within a line and one in four of
 * the 16-byte NEON loads straddles a line boundary. Aligning removes that. */
static void *amalloc(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, n) != 0) return NULL;
    return p;
}

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

#define DREGS (SGM_D / 16)          /* 4 at D=64, 8 at D=128 */

/* One recurrence step over all D disparities.
 * prev/cur: DREGS registers. cost: DREGS registers. Returns nothing; the
 * caller supplies mn (the previous pixel's min) already broadcast. */
static inline void step_neon(const uint8x16_t *prev, const uint8x16_t *cost,
                             uint8x16_t *cur, uint8x16_t mn)
{
    const uint8x16_t p1  = vdupq_n_u8(SGM_P1);
    const uint8x16_t big = vdupq_n_u8(255);
    const uint8x16_t e   = vqaddq_u8(mn, vdupq_n_u8(SGM_P2));

    for (int r = 0; r < DREGS; r++) {
        /* d-1: shift one lane up, pulling in the previous register's top lane
         * (or 255 at d=0 so it can never win the min). */
        uint8x16_t lo = (r == 0) ? vextq_u8(big, prev[0], 15)
                                 : vextq_u8(prev[r - 1], prev[r], 15);
        /* d+1: shift one lane down, pulling in the next register's low lane
         * (or 255 at d=D-1). */
        uint8x16_t hi = (r == DREGS - 1) ? vextq_u8(prev[r], big, 1)
                                         : vextq_u8(prev[r], prev[r + 1], 1);

        uint8x16_t m = prev[r];
        m = vminq_u8(m, vqaddq_u8(lo, p1));
        m = vminq_u8(m, vqaddq_u8(hi, p1));
        m = vminq_u8(m, e);
        /* m >= mn always (prev[r] >= mn), so the subtract cannot underflow. */
        cur[r] = vqaddq_u8(cost[r], vsubq_u8(m, mn));
    }
}

/* min over all D lanes, broadcast. One vminvq per pixel per direction, off the
 * critical dependency chain (the caller computes it from the value it just
 * produced, while the next pixel's loads are already in flight). */
static inline uint8x16_t hmin_bcast(const uint8x16_t *v)
{
    uint8x16_t m = v[0];
    for (int r = 1; r < DREGS; r++) m = vminq_u8(m, v[r]);
    return vdupq_n_u8(vminvq_u8(m));
}

/* Load the D costs for column x into DREGS registers, handling the left border
 * where x-d < 0 (those disparities are invalid and take COST_INVALID). */
static inline void load_cost(const uint64_t *cl, const uint64_t *cr,
                             int x, uint8x16_t *out)
{
    if (x >= SGM_D - 1 + 15) {              /* fully in range: vector path */
        for (int r = 0; r < DREGS; r++) out[r] = ham16_neon(cl, cr, x, r * 16);
    } else {
        uint8_t tmp[SGM_D];
        for (int d = 0; d < SGM_D; d++) tmp[d] = ham_scalar(cl, cr, x, d);
        for (int r = 0; r < DREGS; r++) out[r] = vld1q_u8(tmp + r * 16);
    }
}


/* Non-temporal store of one 64-byte disparity vector group to a plane.
 * The D and U planes are written once, read once, and never reused, so pulling
 * them through the cache only evicts the descriptors and row state that ARE
 * reused. STNP bypasses the cache hierarchy for these lines. Measured, not
 * assumed: see results/ -- if it does not help on a given core it is compiled
 * back out with -DSGM_NT=0. */
#ifndef SGM_NT
#define SGM_NT 1
#endif
static inline void store_plane(uint8_t *p, const uint8x16_t *v)
{
#if SGM_NT && defined(__aarch64__) && (SGM_D % 32) == 0
    /* stnp writes a register PAIR, so this loop needs an even DREGS. At
     * SGM_D=16 (legal) DREGS is 1 and v[r+1] would read out of bounds. Odd
     * DREGS takes the plain-store path instead. */
    for (int r = 0; r < DREGS; r += 2) {
        __asm__ volatile("stnp %q0, %q1, [%2]"
                         :: "w"(v[r]), "w"(v[r + 1]), "r"(p + r * 16)
                         : "memory");
    }
#else
    for (int r = 0; r < DREGS; r++) vst1q_u8(p + r * 16, v[r]);
#endif
}

/* Prefetch the descriptor pair a few pixels ahead. The A55 is in-order, so a
 * load-use stall costs the full latency; measured IPC there was 0.62. */
#ifndef SGM_PF
#define SGM_PF 0   /* MEASURED: prefetch alone is +0.7% WORSE on A55. Off. */
#endif
static inline void pf(const void *a) {
#if SGM_PF
    __builtin_prefetch(a, 0, 0);
#else
    (void)a;
#endif
}

static int a55_run(const uint8_t *left, const uint8_t *right, int W, int H,
                   uint8_t *disp, int threads, sgm_stage_times *t)
{
    (void)threads;
    const size_t N = (size_t)W * H;
    uint64_t *cl = amalloc(N * 8), *cr = amalloc(N * 8);
    /* PLANE LAYOUT: two separate planes, or one interleaved [D|U] per pixel.
     * Interleaving costs pass 1 a strided (constant-stride, prefetcher-friendly,
     * stnp-bypassed) write, and buys pass 2 ONE 128-byte contiguous read stream
     * instead of two 64-byte streams. This is a direct test of the hypothesis
     * that STREAM COUNT -- not latency or alignment -- is what limits the
     * memory-bound A55: ~6 streams/thread x 6 threads vs a prefetcher that
     * tracks 8-16. If it does not help, that hypothesis is wrong. */
#ifndef SGM_ILV
#define SGM_ILV 0
#endif
#if SGM_ILV
    const size_t PSTRIDE = 2 * SGM_D;
    uint8_t *plane  = amalloc(N * PSTRIDE);
    uint8_t *Dplane = plane;
    uint8_t *Uplane = plane + SGM_D;
#else
    const size_t PSTRIDE = SGM_D;
    uint8_t *Dplane = amalloc(N * SGM_D);
    uint8_t *Uplane = amalloc(N * SGM_D);
#endif
    if (!cl || !cr || !Dplane || !Uplane) return -1;

    double t0 = now_ms();
    census_neon(left,  cl, W, H);
    census_neon(right, cr, W, H);
    double t1 = now_ms();

    /* =====================================================================
     * THREE PASSES, EVERY ONE OF THEM PARALLEL.
     *
     * The previous shape (D stored, then L+R+U fused) was correct and BIT-EXACT
     * but its second pass was serial, which caps the whole run at ~1.33x on six
     * cores by Amdahl -- measured 1.27x. The fix is to choose the parallel axis
     * per path direction rather than for the program as a whole:
     *
     *   U and D are VERTICAL   -> serial down a column, independent ACROSS
     *                             columns -> parallelise over columns.
     *   L and R are HORIZONTAL -> serial along a row, independent ACROSS
     *                             rows    -> parallelise over rows.
     *
     * Nothing else can be parallel: that is the real cost of SGM's recurrence
     * and it is the number the report should carry.
     *
     * Price: U must now be materialised too (2 x 133 MB planes instead of 1).
     * We trade ~260 MB of extra streaming traffic for 6-way parallelism, which
     * is the right trade on a core whose L2 is 64 KB -- everything misses L2
     * either way, so the traffic was never going to be free, but the idle cores
     * were pure waste.
     * ===================================================================== */

    /* ---- pass 1: D (bottom->top) and U (top->bottom), parallel over COLUMN
     * BLOCKS.
     *
     * Parallelising over single columns is correct but walks down a column, so
     * consecutive y are W*8 bytes apart and EVERY load misses -- measured 3.3x
     * slower single-threaded than the row-major version it replaced. The axis
     * was right and the access pattern was wrong.
     *
     * So we sweep a BLOCK of BW columns together: for each y the block touches
     * BW*8 contiguous descriptor bytes and BW*SGM_D contiguous plane bytes, so
     * every cache line that is fetched is fully consumed. The per-column
     * recurrence state is BW*SGM_D = 1 KB, which lives in L1 rather than in
     * registers -- that is the deliberate trade: 1 KB of L1 traffic to turn a
     * stride-15KB access pattern into a sequential one.
     * ---- */
    {
#ifndef SGM_BW
#define SGM_BW 192   /* measured optimum on A55: 64KB L2, see results */
#endif
        const int BW = SGM_BW;
#ifdef _OPENMP
    /* Observed team size, from INSIDE the region. omp_get_max_threads() only
     * echoes the request back through the runtime -- the same failure shape as
     * the -t flag that was found to be inert. */
    if (t) { t->threads_used = 0; t->transfer_ms = -1; }
#ifdef _OPENMP
#pragma omp parallel
    { if (t && omp_get_thread_num() == 0) t->threads_used = omp_get_num_threads(); }
#endif
#pragma omp parallel
#endif
        {
            uint8_t *st = amalloc((size_t)BW * SGM_D);
            uint8x16_t cur[DREGS], c[DREGS], pv[DREGS];
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
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
                        uint8_t *op = Dplane + ((size_t)y * W + x) * PSTRIDE;
                        for (int r = 0; r < DREGS; r++) vst1q_u8(sp + r * 16, cur[r]);
                        store_plane(op, cur);
                        if (y > 8) pf(cl + (size_t)(y - 8) * W + x);
                    }
                }
            }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int xb = 0; xb < W; xb += BW) {
                const int xe = (xb + BW < W) ? xb + BW : W;
                for (int y = 0; y < H; y++) {
                    const uint64_t *rl = cl + (size_t)y * W, *rr = cr + (size_t)y * W;
                    for (int x = xb; x < xe; x++) {
                        uint8_t *sp = st + (size_t)(x - xb) * SGM_D;
                        load_cost(rl, rr, x, c);
                        if (y == 0) for (int r = 0; r < DREGS; r++) cur[r] = c[r];
                        else {
                            for (int r = 0; r < DREGS; r++) pv[r] = vld1q_u8(sp + r * 16);
                            step_neon(pv, c, cur, hmin_bcast(pv));
                        }
                        uint8_t *op = Uplane + ((size_t)y * W + x) * PSTRIDE;
                        for (int r = 0; r < DREGS; r++) vst1q_u8(sp + r * 16, cur[r]);
                        store_plane(op, cur);
                        if (y + 8 < H) pf(cl + (size_t)(y + 8) * W + x);
                    }
                }
            }
            free(st);
        }
    }

    /* ---- pass 2: L and R on the fly + sum + argmin, parallel over rows ---- */
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        uint8_t *rowR = amalloc((size_t)W * SGM_D);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int y = 0; y < H; y++) {
            const uint64_t *rl = cl + (size_t)y * W, *rr = cr + (size_t)y * W;
            uint8x16_t pv[DREGS], cur[DREGS], c[DREGS];

            for (int x = W - 1; x >= 0; x--) {           /* R: right -> left */
                load_cost(rl, rr, x, c);
                if (x == W - 1) for (int r = 0; r < DREGS; r++) cur[r] = c[r];
                else            step_neon(pv, c, cur, hmin_bcast(pv));
                for (int r = 0; r < DREGS; r++) {
                    vst1q_u8(rowR + (size_t)x * SGM_D + r * 16, cur[r]);
                    pv[r] = cur[r];
                }
            }

            uint8x16_t pvL[DREGS], curL[DREGS];
            for (int x = 0; x < W; x++) {                /* L: left -> right */
                load_cost(rl, rr, x, c);
                if (x == 0) for (int r = 0; r < DREGS; r++) curL[r] = c[r];
                else        step_neon(pvL, c, curL, hmin_bcast(pvL));

                const uint8_t *dp = Dplane + ((size_t)y * W + x) * PSTRIDE;
                const uint8_t *up = Uplane + ((size_t)y * W + x) * PSTRIDE;
                const uint8_t *Rb = rowR + (size_t)x * SGM_D;
                uint16x8_t sum[DREGS * 2];
                for (int r = 0; r < DREGS; r++) {
                    uint8x16_t Rv = vld1q_u8(Rb + r * 16);
                    uint8x16_t Dv = vld1q_u8(dp + r * 16);
                    uint8x16_t Uv = vld1q_u8(up + r * 16);
                    uint16x8_t lo = vaddl_u8(vget_low_u8(curL[r]), vget_low_u8(Uv));
                    uint16x8_t hi = vaddl_high_u8(curL[r], Uv);
                    lo = vaddq_u16(lo, vaddl_u8(vget_low_u8(Rv), vget_low_u8(Dv)));
                    hi = vaddq_u16(hi, vaddl_high_u8(Rv, Dv));
                    sum[r * 2] = lo; sum[r * 2 + 1] = hi;
                    pvL[r] = curL[r];
                }
                uint16x8_t mv = sum[0];
                for (int k = 1; k < DREGS * 2; k++) mv = vminq_u16(mv, sum[k]);
                uint16_t best = vminvq_u16(mv);
                uint16x8_t bv = vdupq_n_u16(best);
                int bestd = 0;
                for (int k = 0; k < DREGS * 2; k++) {
                    uint8x8_t n = vmovn_u16(vceqq_u16(sum[k], bv));
                    uint64_t m = vget_lane_u64(vreinterpret_u64_u8(n), 0);
                    if (m) { bestd = k * 8 + (__builtin_ctzll(m) >> 3); break; }
                }
                disp[(size_t)y * W + x] = (uint8_t)bestd;
            }
        }
        free(rowR);
    }
    double t2 = now_ms();

    if (t) { t->census_ms = t1 - t0; t->cost_ms = -1;
             t->aggregate_ms = t2 - t1; t->argmin_ms = -1; }
    #if SGM_ILV
    free(cl); free(cr); free(plane);
#else
    free(cl); free(cr); free(Dplane); free(Uplane);
#endif
    return 0;
}

const sgm_impl SGM_IMPL = { "a55", a55_run };

#else

/* Non-temporal store of one 64-byte disparity vector group to a plane.
 * The D and U planes are written once, read once, and never reused, so pulling
 * them through the cache only evicts the descriptors and row state that ARE
 * reused. STNP bypasses the cache hierarchy for these lines. Measured, not
 * assumed: see results/ -- if it does not help on a given core it is compiled
 * back out with -DSGM_NT=0. */
#ifndef SGM_NT
#define SGM_NT 1
#endif
static inline void store_plane(uint8_t *p, const uint8x16_t *v)
{
#if SGM_NT && defined(__aarch64__) && (SGM_D % 32) == 0
    /* stnp writes a register PAIR, so this loop needs an even DREGS. At
     * SGM_D=16 (legal) DREGS is 1 and v[r+1] would read out of bounds. Odd
     * DREGS takes the plain-store path instead. */
    for (int r = 0; r < DREGS; r += 2) {
        __asm__ volatile("stnp %q0, %q1, [%2]"
                         :: "w"(v[r]), "w"(v[r + 1]), "r"(p + r * 16)
                         : "memory");
    }
#else
    for (int r = 0; r < DREGS; r++) vst1q_u8(p + r * 16, v[r]);
#endif
}

/* Prefetch the descriptor pair a few pixels ahead. The A55 is in-order, so a
 * load-use stall costs the full latency; measured IPC there was 0.62. */
#ifndef SGM_PF
#define SGM_PF 0   /* MEASURED: prefetch alone is +0.7% WORSE on A55. Off. */
#endif
static inline void pf(const void *a) {
#if SGM_PF
    __builtin_prefetch(a, 0, 0);
#else
    (void)a;
#endif
}

static int a55_run(const uint8_t *l, const uint8_t *r, int W, int H,
                   uint8_t *d, int th, sgm_stage_times *t)
{ (void)l;(void)r;(void)W;(void)H;(void)d;(void)th;(void)t; return -1; }
const sgm_impl SGM_IMPL = { "a55(no-neon)", a55_run };
#endif
