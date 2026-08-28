/* gen_synthetic.c — deterministic textured stereo pair with known disparity.
 *
 *   gen_synthetic W H OUTDIR [seed]
 * writes OUTDIR/left.pgm, right.pgm, gt.pgm
 *
 * Scene: a textured background plane at small disparity plus a few textured
 * rectangular slabs at larger disparities (so there are real depth edges and
 * occlusions). Texture is band-limited noise so census has something to bite
 * on without being pure white noise. Convention: left(x) = right(x - d).
 * Occluded / unmatched regions in left are filled from the background.
 *
 * This is for correctness and cache behaviour only. The real pair from
 * Kyle's cameras is the one that matters for the final numbers.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "sgm_params.h"

static uint32_t rng;
static uint32_t xr(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

/* smooth-ish texture: 3 octaves of blurred noise */
static void texture(uint8_t *im, int W, int H, uint32_t seed) {
    rng = seed ? seed : 1;
    float *a = calloc((size_t)W * H, sizeof(float));
    for (int oct = 0, cell = 2; oct < 3; oct++, cell *= 4) {
        int gw = W / cell + 2, gh = H / cell + 2;
        float *g = malloc(sizeof(float) * gw * gh);
        for (int i = 0; i < gw * gh; i++) g[i] = (xr() & 0xffff) / 65535.0f;
        float amp = 1.0f / (1 << oct);
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            float fx = (float)x / cell, fy = (float)y / cell;
            int ix = (int)fx, iy = (int)fy; float tx = fx - ix, ty = fy - iy;
            float v = g[iy * gw + ix] * (1 - tx) * (1 - ty) + g[iy * gw + ix + 1] * tx * (1 - ty)
                    + g[(iy + 1) * gw + ix] * (1 - tx) * ty + g[(iy + 1) * gw + ix + 1] * tx * ty;
            a[y * W + x] += v * amp;
        }
        free(g);
    }
    for (int i = 0; i < W * H; i++) { float v = a[i] / 1.75f * 255.0f; im[i] = v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; }
    free(a);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s W H OUTDIR [seed]\n", argv[0]); return 1; }
    int W = atoi(argv[1]), H = atoi(argv[2]);
    const char *out = argv[3];
    uint32_t seed = argc > 4 ? (uint32_t)atoi(argv[4]) : 1;

    uint8_t *right = malloc((size_t)W * H), *left = malloc((size_t)W * H), *gt = malloc((size_t)W * H);
    uint8_t *slab = malloc((size_t)W * H);
    texture(right, W, H, seed);
    texture(slab, W, H, seed * 7919 + 17);

    /* ground-truth disparity in RIGHT-image coordinates first */
    uint8_t *gtr = malloc((size_t)W * H);
    int bgd = SGM_D / 8;
    memset(gtr, bgd, (size_t)W * H);
    rng = seed * 31 + 5;
    int nslab = 6;
    for (int s = 0; s < nslab; s++) {
        int sw = W / 6 + (xr() % (W / 4)), sh = H / 6 + (xr() % (H / 4));
        int sx = xr() % (W - sw), sy = xr() % (H - sh);
        int d = SGM_D / 4 + (xr() % (SGM_D / 2)); if (d > SGM_D - 2) d = SGM_D - 2;
        for (int y = sy; y < sy + sh; y++) for (int x = sx; x < sx + sw; x++) {
            if (d > gtr[y * W + x]) { gtr[y * W + x] = d; right[y * W + x] = slab[y * W + x]; }
        }
    }
    /* build LEFT: left(x + d) = right(x), nearer (larger d) wins; fill holes from background */
    memset(gt, 0, (size_t)W * H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) left[y * W + x] = 0;
        for (int x = 0; x < W; x++) {
            int d = gtr[y * W + x], xl = x + d;
            if (xl < W && d >= gt[y * W + xl]) { left[y * W + xl] = right[y * W + x]; gt[y * W + xl] = d; }
        }
        for (int x = 0; x < W; x++) if (gt[y * W + x] == 0) {  /* hole: background */
            int xr_ = x - bgd; if (xr_ < 0) xr_ = 0;
            left[y * W + x] = right[y * W + xr_]; gt[y * W + x] = bgd;
        }
    }

    char p[1024];
    FILE *f;
#define WRITE(name, buf) do { snprintf(p, sizeof p, "%s/%s", out, name); f = fopen(p, "wb"); \
    if (!f) { fprintf(stderr, "cannot write %s\n", p); return 1; } \
    fprintf(f, "P5\n%d %d\n255\n", W, H); fwrite(buf, 1, (size_t)W * H, f); fclose(f); } while (0)
    WRITE("left.pgm", left); WRITE("right.pgm", right); WRITE("gt.pgm", gt);
    printf("wrote %s/{left,right,gt}.pgm %dx%d seed %u (bg d=%d, %d slabs)\n", out, W, H, seed, bgd, nslab);
    return 0;
}
