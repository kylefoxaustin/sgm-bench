/* census_neon.h — NEON 9x7 (SGM_CENSUS_W x SGM_CENSUS_H) census transform.
 *
 * Produces the SAME uint64 descriptors as the scalar census in sgm_ref.c:
 * bit n set iff neighbour < centre, n incrementing dy-outer / dx-inner and
 * skipping the centre, edge-replicated at the borders.
 *
 * NOTE ON BIT ORDER (a freedom we are deliberately NOT taking yet):
 * the matching cost is popcount(a ^ b), so any permutation of the bits applied
 * consistently to BOTH descriptors leaves every Hamming distance -- and
 * therefore every argmin, and therefore the whole disparity map -- unchanged.
 * A future packing may reorder bits for speed. This one reproduces the scalar
 * order exactly so that the descriptors themselves can be diffed during
 * bring-up, which is a strictly stronger test than diffing the output.
 */
#ifndef CENSUS_NEON_H_INCLUDED
#define CENSUS_NEON_H_INCLUDED

#include <stdint.h>

/* im: W*H greyscale. out: W*H descriptors. Threads via OpenMP if enabled. */
void census_neon(const uint8_t *im, uint64_t *out, int W, int H);

/* Scalar twin, byte-identical semantics, for the bring-up self-test. */
void census_scalar_ref(const uint8_t *im, uint64_t *out, int W, int H);

#endif
