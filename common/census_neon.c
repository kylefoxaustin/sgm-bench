/* census_neon.c — NEON 9x7 census transform.
 *
 * Shape of the problem: SGM_CENSUS_BITS (62 for 9x7) comparisons per pixel,
 * each producing one bit. The awkward part is not the comparisons -- those
 * vectorise perfectly, 16 pixels per vcltq_u8 -- it is getting 62 bits per
 * pixel out of a lane-parallel layout and into one uint64 per pixel.
 *
 * Strategy:
 *   1. Accumulate into 8 "byte planes": plane b of a uint8x16_t holds byte b
 *      of the descriptor for 16 consecutive pixels. Comparison n contributes
 *      bit (n%8) of plane (n/8). All 62 comparisons stay in registers.
 *   2. Transpose 8 planes x 16 pixels -> 16 uint64 with two ST4 stores (which
 *      give a 4-way byte interleave for free) plus four zip pairs. That is
 *      ~24 ops per 16 pixels, i.e. 1.5 ops/pixel of packing overhead on top
 *      of the 62 compares -- small enough not to dominate.
 *
 * Borders (where the 9x7 window leaves the image) are done scalar with edge
 * replication, matching the oracle. The interior is the vector path.
 */
#include "census_neon.h"
#include "sgm_params.h"
#include <string.h>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define HAVE_NEON 1
#endif

#define RX (SGM_CENSUS_W / 2)   /* 4 */
#define RY (SGM_CENSUS_H / 2)   /* 3 */
#define NBITS SGM_CENSUS_BITS   /* 62 */

static inline int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Scalar twin: identical semantics to sgm_ref.c's census(). */
void census_scalar_ref(const uint8_t *im, uint64_t *out, int W, int H)
{
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t c = im[y * W + x];
            uint64_t s = 0;
            int n = 0;
            for (int dy = -RY; dy <= RY; dy++) {
                int yy = clampi(y + dy, 0, H - 1);
                for (int dx = -RX; dx <= RX; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int xx = clampi(x + dx, 0, W - 1);
                    if (im[yy * W + xx] < c) s |= (uint64_t)1 << n;
                    n++;
                }
            }
            out[y * W + x] = s;
        }
    }
}

#ifdef HAVE_NEON

/* Byte offsets (relative to the centre pixel) of the 62 window taps, in the
 * oracle's order: dy outer, dx inner, centre skipped. Built once per image
 * because the row stride W is a runtime value. */
static void build_taps(int W, int *tap)
{
    int n = 0;
    for (int dy = -RY; dy <= RY; dy++)
        for (int dx = -RX; dx <= RX; dx++) {
            if (dx == 0 && dy == 0) continue;
            tap[n++] = dy * W + dx;
        }
}

void census_neon(const uint8_t *im, uint64_t *out, int W, int H)
{
    int tap[NBITS];
    build_taps(W, tap);

    /* Border rows/cols: scalar, edge-replicated, matching the oracle. The
     * interior loop below never reads out of bounds, so it needs no clamp. */
    const int x0 = RX, x1 = W - RX;          /* [x0, x1) has a full window */
    const int y0 = RY, y1 = H - RY;

    if (W < SGM_CENSUS_W || H < SGM_CENSUS_H) {   /* degenerate: all border */
        census_scalar_ref(im, out, W, H);
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < H; y++) {
        if (y < y0 || y >= y1) {
            /* full scalar row */
            for (int x = 0; x < W; x++) {
                uint8_t c = im[y * W + x];
                uint64_t s = 0; int n = 0;
                for (int dy = -RY; dy <= RY; dy++) {
                    int yy = clampi(y + dy, 0, H - 1);
                    for (int dx = -RX; dx <= RX; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        int xx = clampi(x + dx, 0, W - 1);
                        if (im[yy * W + xx] < c) s |= (uint64_t)1 << n;
                        n++;
                    }
                }
                out[y * W + x] = s;
            }
            continue;
        }

        /* left border of an interior row */
        for (int x = 0; x < x0; x++) {
            uint8_t c = im[y * W + x];
            uint64_t s = 0; int n = 0;
            for (int dy = -RY; dy <= RY; dy++)
                for (int dx = -RX; dx <= RX; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int xx = clampi(x + dx, 0, W - 1);
                    if (im[(y + dy) * W + xx] < c) s |= (uint64_t)1 << n;
                    n++;
                }
            out[y * W + x] = s;
        }

        /* ---- vector interior ---- */
        const uint8x16_t one = vdupq_n_u8(1);
        int x = x0;
        for (; x + 16 <= x1; x += 16) {
            const uint8_t *p = im + (size_t)y * W + x;
            const uint8x16_t c = vld1q_u8(p);
            uint8x16_t a0, a1, a2, a3, a4, a5, a6, a7;
            uint8x16_t nb, m;

            /* One comparison: bit j of the plane currently being built.
             * vcltq_u8(nb, c) is 0xFF where neighbour < centre, matching the
             * oracle's predicate exactly. */
#define TAP(ACC, J, N)                                                        \
            nb  = vld1q_u8(p + tap[(N)]);                                     \
            m   = vcltq_u8(nb, c);                                            \
            ACC = vorrq_u8(ACC, vshlq_n_u8(vandq_u8(m, one), (J)))
#define PLANE(ACC, BASE)                                                      \
            ACC = vdupq_n_u8(0);                                              \
            TAP(ACC, 0, (BASE) + 0); TAP(ACC, 1, (BASE) + 1);                 \
            TAP(ACC, 2, (BASE) + 2); TAP(ACC, 3, (BASE) + 3);                 \
            TAP(ACC, 4, (BASE) + 4); TAP(ACC, 5, (BASE) + 5);                 \
            TAP(ACC, 6, (BASE) + 6); TAP(ACC, 7, (BASE) + 7)

            PLANE(a0,  0); PLANE(a1,  8); PLANE(a2, 16); PLANE(a3, 24);
            PLANE(a4, 32); PLANE(a5, 40); PLANE(a6, 48);
            /* last plane is short: 62 taps means bits 56..61 only */
            a7 = vdupq_n_u8(0);
            TAP(a7, 0, 56); TAP(a7, 1, 57); TAP(a7, 2, 58);
            TAP(a7, 3, 59); TAP(a7, 4, 60); TAP(a7, 5, 61);
#undef PLANE
#undef TAP

            /* Transpose 8 planes x 16 pixels -> 16 uint64.
             * ST4 gives a 4-way byte interleave, so lo[] ends up holding
             * bytes 0..3 of each descriptor and hi[] bytes 4..7; one zip pair
             * per 4 pixels then fuses them into uint64 lanes. */
            uint8_t lo[64], hi[64];
            uint8x16x4_t q;
            q.val[0] = a0; q.val[1] = a1; q.val[2] = a2; q.val[3] = a3;
            vst4q_u8(lo, q);
            q.val[0] = a4; q.val[1] = a5; q.val[2] = a6; q.val[3] = a7;
            vst4q_u8(hi, q);

            uint64_t *o = out + (size_t)y * W + x;
            for (int k = 0; k < 4; k++) {
                uint32x4_t l = vld1q_u32((const uint32_t *)(lo + 16 * k));
                uint32x4_t h = vld1q_u32((const uint32_t *)(hi + 16 * k));
                vst1q_u64(o + 4 * k,
                          vreinterpretq_u64_u32(vzip1q_u32(l, h)));
                vst1q_u64(o + 4 * k + 2,
                          vreinterpretq_u64_u32(vzip2q_u32(l, h)));
            }
        }

        /* vector tail + right border, scalar */
        for (; x < W; x++) {
            uint8_t c = im[y * W + x];
            uint64_t s = 0; int n = 0;
            for (int dy = -RY; dy <= RY; dy++)
                for (int dx = -RX; dx <= RX; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int xx = clampi(x + dx, 0, W - 1);
                    if (im[(y + dy) * W + xx] < c) s |= (uint64_t)1 << n;
                    n++;
                }
            out[y * W + x] = s;
        }
    }
}

#else  /* no NEON: fall back so the file still builds and tests on x86 */

void census_neon(const uint8_t *im, uint64_t *out, int W, int H)
{
    census_scalar_ref(im, out, W, H);
}

#endif
