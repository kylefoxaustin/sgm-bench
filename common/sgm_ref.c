/* sgm_ref.c — THE ORACLE.
 *
 * Plain C, single thread, no NEON, no OpenMP, no tricks. Materialises the
 * full cost volume (allowed ONLY here). Slow is fine. Its output is the
 * definition of "correct" for every other implementation in this repo.
 *
 * Do not optimise this file. If you think it's wrong, fix sgm_params.h's
 * semantics comment and this file together, then regenerate golden.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sgm.h"

#define D SGM_D

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ---- census: SGM_CENSUS_W x SGM_CENSUS_H, edge-replicated ---- */
static void census(const uint8_t *im, uint64_t *out, int W, int H) {
    const int rx = SGM_CENSUS_W / 2, ry = SGM_CENSUS_H / 2;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint8_t c = im[y * W + x];
            uint64_t s = 0; int n = 0;
            for (int dy = -ry; dy <= ry; dy++) {
                int yy = clampi(y + dy, 0, H - 1);
                for (int dx = -rx; dx <= rx; dx++) {
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

/* ---- Hamming cost volume C[(y*W+x)*D + d] ---- */
static void cost_volume(const uint64_t *cl, const uint64_t *cr, uint8_t *C, int W, int H) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint64_t a = cl[y * W + x];
            uint8_t *c = C + ((size_t)y * W + x) * D;
            for (int d = 0; d < D; d++) {
                int xr = x - d;
                c[d] = (xr < 0) ? SGM_COST_INVALID
                                : (uint8_t)__builtin_popcountll(a ^ cr[y * W + xr]);
            }
        }
    }
}

static inline int sat255(int v) { return v > 255 ? 255 : v; }

/* one recurrence step: prev -> cur given cost c */
static void step(const uint8_t *prev, const uint8_t *c, uint8_t *cur) {
    int mn = 255;
    for (int d = 0; d < D; d++) if (prev[d] < mn) mn = prev[d];
    int e = sat255(mn + SGM_P2);
    for (int d = 0; d < D; d++) {
        int a = prev[d];
        int b = (d > 0)     ? sat255(prev[d - 1] + SGM_P1) : 255;
        int f = (d < D - 1) ? sat255(prev[d + 1] + SGM_P1) : 255;
        int m = a; if (b < m) m = b; if (f < m) m = f; if (e < m) m = e;
        cur[d] = (uint8_t)(c[d] + m - mn);   /* <= COST_MAX + P2 <= 255 by construction */
    }
}

/* aggregate one direction (dx,dy) into S */
static void aggregate_dir(const uint8_t *C, uint16_t *S, int W, int H, int dx, int dy,
                          uint8_t *Lprev, uint8_t *Lcur) {
    int y0 = dy < 0 ? H - 1 : 0, y1 = dy < 0 ? -1 : H, ys = dy < 0 ? -1 : 1;
    int x0 = dx < 0 ? W - 1 : 0, x1 = dx < 0 ? -1 : W, xs = dx < 0 ? -1 : 1;
    for (int y = y0; y != y1; y += ys) {
        for (int x = x0; x != x1; x += xs) {
            const uint8_t *c = C + ((size_t)y * W + x) * D;
            uint8_t *cur = Lcur + (size_t)x * D;
            int px = x - dx, py = y - dy;
            const uint8_t *prev = NULL;
            if (px >= 0 && px < W && py >= 0 && py < H)
                prev = (dy == 0) ? Lcur + (size_t)px * D : Lprev + (size_t)px * D;
            if (prev) step(prev, c, cur);
            else      memcpy(cur, c, D);
            uint16_t *s = S + ((size_t)y * W + x) * D;
            for (int d = 0; d < D; d++) s[d] += cur[d];
        }
        uint8_t *t = Lprev; Lprev = Lcur; Lcur = t;   /* row swap */
    }
}

static const int DIRS8[8][2] = {
    { 1, 0}, {-1, 0}, { 0, 1}, { 0,-1},          /* L->R, R->L, T->B, B->T */
    { 1, 1}, {-1, 1}, { 1,-1}, {-1,-1}           /* diagonals */
};

static int ref_run(const uint8_t *left, const uint8_t *right, int W, int H,
                   uint8_t *disp, int threads, sgm_stage_times *t) {
    (void)threads;
    size_t N = (size_t)W * H;
    uint64_t *cl = malloc(N * 8), *cr = malloc(N * 8);
    uint8_t  *C  = malloc(N * D);
    uint16_t *S  = calloc(N * D, 2);
    uint8_t  *La = malloc((size_t)W * D), *Lb = malloc((size_t)W * D);
    if (!cl || !cr || !C || !S || !La || !Lb) { fprintf(stderr, "ref: alloc failed\n"); return -1; }

    double t0 = now_ms();
    census(left, cl, W, H);
    census(right, cr, W, H);
    double t1 = now_ms();
    cost_volume(cl, cr, C, W, H);
    double t2 = now_ms();
    for (int p = 0; p < SGM_PATHS; p++)
        aggregate_dir(C, S, W, H, DIRS8[p][0], DIRS8[p][1], La, Lb);
    double t3 = now_ms();
    for (size_t i = 0; i < N; i++) {
        const uint16_t *s = S + i * D;
        int best = 0; uint16_t bv = s[0];
        for (int d = 1; d < D; d++) if (s[d] < bv) { bv = s[d]; best = d; }  /* strict: lowest d wins ties */
        disp[i] = (uint8_t)best;
    }
    double t4 = now_ms();

    if (t) { t->census_ms = t1 - t0; t->cost_ms = t2 - t1; t->aggregate_ms = t3 - t2; t->argmin_ms = t4 - t3; }
    free(cl); free(cr); free(C); free(S); free(La); free(Lb);
    return 0;
}

const sgm_impl SGM_IMPL = { "ref", ref_run };
