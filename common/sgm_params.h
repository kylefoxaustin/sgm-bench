/* sgm_params.h — THE ONLY PLACE the workload parameters live.
 *
 * Every implementation (ref, a55, a720, mali_cl, cuda_libsgm) includes this
 * and must not hardcode any of these values. Change here, rebuild everything,
 * regenerate golden, re-run everything. No exceptions.
 *
 * Image width/height are runtime (read from the PGM), not compile-time.
 */
#ifndef SGM_PARAMS_H
#define SGM_PARAMS_H

/* ---- Workload configuration. These are the real, benchmarked values. ---- */
#ifndef SGM_D
#define SGM_D          64      /* disparity range, must be a multiple of 16 */
#endif
#ifndef SGM_PATHS
#define SGM_PATHS      4       /* 4 = L,R,U,D   8 = + the four diagonals   */
#endif
#define SGM_CENSUS_W   9       /* census window width  (odd)                */
#define SGM_CENSUS_H   7       /* census window height (odd)                */

/* ---- Penalties. 8-bit path cost requires SGM_COST_MAX + SGM_P2 <= 255 ---- */
#define SGM_P1         20
#define SGM_P2         192

/* ---- Derived / fixed ---- */
#define SGM_CENSUS_BITS  (SGM_CENSUS_W * SGM_CENSUS_H - 1)   /* 62 for 9x7 (centre skipped) */
#define SGM_COST_MAX     SGM_CENSUS_BITS                     /* Hamming max */
#define SGM_COST_INVALID SGM_COST_MAX   /* cost where x-d < 0 (no match)   */

#if SGM_CENSUS_BITS > 64
#error "census window exceeds 64 bits; descriptor type must change"
#endif
#if (SGM_COST_MAX + SGM_P2) > 255
#error "SGM_COST_MAX + SGM_P2 must be <= 255 for 8-bit path costs"
#endif
#if (SGM_D % 16) != 0
#error "SGM_D must be a multiple of 16"
#endif
#if SGM_PATHS != 4 && SGM_PATHS != 8
#error "SGM_PATHS must be 4 or 8"
#endif

/* Semantics, fixed for all implementations:
 *  - Census: bit n set iff neighbour < centre. n increments over dy (outer,
 *    top to bottom) then dx (inner, left to right), skipping the centre.
 *    Out-of-image neighbours are edge-replicated (clamped coordinates).
 *  - Cost:   C(x,y,d) = popcount(censusL(x,y) ^ censusR(x-d,y)) if x-d >= 0,
 *            else SGM_COST_INVALID.
 *  - Path cost L (uint8) with renormalisation:
 *            L(p,d) = C(p,d) + min(Lp(d), Lp(d-1)+P1, Lp(d+1)+P1, min_k Lp(k)+P2)
 *                     - min_k Lp(k)
 *            where Lp = L at the previous pixel along the path; additions
 *            saturate at 255; d-1 / d+1 out of range count as 255.
 *            First pixel on a path: L = C.
 *  - Sum S (uint16) over the SGM_PATHS directions.
 *  - Disparity = argmin_d S, lowest d wins ties. Output uint8.
 *  - No LR check, no subpixel, no median.
 */

#endif /* SGM_PARAMS_H */
