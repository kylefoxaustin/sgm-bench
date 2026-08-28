/* sgm.h — interface every implementation exposes to the harness. */
#ifndef SGM_H
#define SGM_H
#include <stdint.h>
#include <stddef.h>
#include "sgm_params.h"

/* Per-stage timings, filled by the implementation if it can (ms). -1 = n/a. */
typedef struct {
    double census_ms;
    double cost_ms;       /* -1 if cost is fused into aggregation */
    double aggregate_ms;
    double argmin_ms;     /* -1 if fused */
} sgm_stage_times;

/* Run one frame. left/right are W*H uint8 rectified grey. disp is W*H uint8
 * out. threads: 0 = implementation default. Returns 0 on success. */
typedef int (*sgm_run_fn)(const uint8_t *left, const uint8_t *right,
                          int W, int H, uint8_t *disp, int threads,
                          sgm_stage_times *t);

typedef struct {
    const char *name;
    sgm_run_fn  run;
} sgm_impl;

/* Each implementation defines one of these. The harness is linked against
 * exactly one implementation per binary. */
extern const sgm_impl SGM_IMPL;

/* ---- PGM (P5, 8-bit) ---- */
uint8_t *pgm_read(const char *path, int *W, int *H);   /* malloc'd, NULL on error */
int      pgm_write(const char *path, const uint8_t *img, int W, int H);

/* ---- 64-bit FNV-1a, used for quick equality checks; scripts also
 *      sha256sum the written PGM for the record. ---- */
uint64_t fnv1a64(const void *data, size_t n);

/* ---- monotonic clock in ms ---- */
double now_ms(void);

#endif
