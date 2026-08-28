/* util.c — PGM I/O, FNV-1a, clock. No external dependencies. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "sgm.h"

static int skip_ws_comments(FILE *f) {
    int c;
    for (;;) {
        c = fgetc(f);
        if (c == '#') { while (c != '\n' && c != EOF) c = fgetc(f); }
        else if (!isspace(c)) { ungetc(c, f); return 0; }
        if (c == EOF) return -1;
    }
}

uint8_t *pgm_read(const char *path, int *W, int *H) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "pgm_read: cannot open %s\n", path); return NULL; }
    char magic[3] = {0};
    if (fread(magic, 1, 2, f) != 2 || magic[0] != 'P' || magic[1] != '5') {
        fprintf(stderr, "pgm_read: %s is not binary PGM (P5)\n", path); fclose(f); return NULL;
    }
    int w, h, maxv;
    if (skip_ws_comments(f) || fscanf(f, "%d", &w) != 1 ||
        skip_ws_comments(f) || fscanf(f, "%d", &h) != 1 ||
        skip_ws_comments(f) || fscanf(f, "%d", &maxv) != 1) {
        fprintf(stderr, "pgm_read: bad header in %s\n", path); fclose(f); return NULL;
    }
    fgetc(f); /* single whitespace after maxval */
    if (maxv != 255) { fprintf(stderr, "pgm_read: %s maxval %d, need 255\n", path, maxv); fclose(f); return NULL; }
    size_t n = (size_t)w * h;
    uint8_t *img = malloc(n);
    if (!img || fread(img, 1, n, f) != n) {
        fprintf(stderr, "pgm_read: short read in %s\n", path); free(img); fclose(f); return NULL;
    }
    fclose(f);
    *W = w; *H = h;
    return img;
}

int pgm_write(const char *path, const uint8_t *img, int W, int H) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "pgm_write: cannot open %s\n", path); return -1; }
    fprintf(f, "P5\n%d %d\n255\n", W, H);
    size_t n = (size_t)W * H;
    int ok = fwrite(img, 1, n, f) == n;
    fclose(f);
    return ok ? 0 : -1;
}

uint64_t fnv1a64(const void *data, size_t n) {
    const uint8_t *p = data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}
