/* hamming_neon.h — matching cost, as an INLINE primitive.
 *
 * Deliberately a header-only inline and not a pass: rule 3 forbids
 * materialising the W x H x D cost volume, so the aggregation sweep calls this
 * per pixel and consumes the 16 costs in-register. 1080p x D=64 would be 133 MB
 * written and re-read; the descriptors it reads instead are 2 x 1080p x 8 B.
 *
 * THE TRICK THAT MAKES IT CHEAP:
 * the naive form reduces each 64-bit XOR to a scalar popcount separately,
 * which costs a 3-deep vpaddl chain PER DISPARITY. Instead we keep 16
 * disparities in flight and reduce ACROSS them with a 3-level vpaddq_u8 tree:
 *   16 descriptors = 8 vectors -> 4 -> 2 -> 1 vector of 16 byte-sized counts.
 * That is 8 veor + 8 vcnt + 7 vpaddq + 1 reverse = 24 ops per 16 disparities,
 * i.e. 1.5 ops/disparity instead of ~3. Max popcount is SGM_CENSUS_BITS (62),
 * so the running sums never leave uint8 and no widening is needed.
 */
#ifndef HAMMING_NEON_H_INCLUDED
#define HAMMING_NEON_H_INCLUDED

#include <stdint.h>
#include "sgm_params.h"

/* Scalar twin. cost for disparity d at column x on one row. */
static inline uint8_t ham_scalar(const uint64_t *cl, const uint64_t *cr,
                                 int x, int d)
{
    int xr = x - d;
    if (xr < 0) return (uint8_t)SGM_COST_INVALID;
    return (uint8_t)__builtin_popcountll(cl[x] ^ cr[xr]);
}

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

/* 16 costs for disparities d0..d0+15 at column x, lane i = disparity d0+i.
 * REQUIRES x - d0 - 15 >= 0; the caller handles the left border where some
 * disparities are invalid (that region is a thin strip and is done scalar). */
static inline uint8x16_t ham16_neon(const uint64_t *cl, const uint64_t *cr,
                                    int x, int d0)
{
    const uint64x2_t a = vdupq_n_u64(cl[x]);
    const uint64_t *p = cr + (x - d0 - 15);   /* ascending: d0+15 .. d0 */

    /* 8 vectors = 16 descriptors; xor then per-byte popcount */
    uint8x16_t c0 = vcntq_u8(vreinterpretq_u8_u64(veorq_u64(a, vld1q_u64(p + 0))));
    uint8x16_t c1 = vcntq_u8(vreinterpretq_u8_u64(veorq_u64(a, vld1q_u64(p + 2))));
    uint8x16_t c2 = vcntq_u8(vreinterpretq_u8_u64(veorq_u64(a, vld1q_u64(p + 4))));
    uint8x16_t c3 = vcntq_u8(vreinterpretq_u8_u64(veorq_u64(a, vld1q_u64(p + 6))));
    uint8x16_t c4 = vcntq_u8(vreinterpretq_u8_u64(veorq_u64(a, vld1q_u64(p + 8))));
    uint8x16_t c5 = vcntq_u8(vreinterpretq_u8_u64(veorq_u64(a, vld1q_u64(p +10))));
    uint8x16_t c6 = vcntq_u8(vreinterpretq_u8_u64(veorq_u64(a, vld1q_u64(p +12))));
    uint8x16_t c7 = vcntq_u8(vreinterpretq_u8_u64(veorq_u64(a, vld1q_u64(p +14))));

    /* 3-level pairwise tree: 8 vectors -> 1, each lane the sum of one
     * descriptor's 8 bytes. Sums stay <= 62, so uint8 is safe throughout. */
    uint8x16_t d0_ = vpaddq_u8(c0, c1), d1_ = vpaddq_u8(c2, c3);
    uint8x16_t d2_ = vpaddq_u8(c4, c5), d3_ = vpaddq_u8(c6, c7);
    uint8x16_t e0  = vpaddq_u8(d0_, d1_), e1 = vpaddq_u8(d2_, d3_);
    uint8x16_t f   = vpaddq_u8(e0, e1);

    /* f is in ascending ADDRESS order = descending disparity; flip so that
     * lane i is disparity d0+i. */
    static const uint8_t rev[16] = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
    return vqtbl1q_u8(f, vld1q_u8(rev));
}
#endif /* NEON */

#endif
