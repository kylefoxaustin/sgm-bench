/* cpu_bandwidth.c — the CPU-side streaming-copy ceiling, mirror of
 * tools/gpu_bandwidth.cu. One number per BOARD: on these SoCs the CPU, Mali
 * and NPU share a single LPDDR bus, so this ceiling serves every row measured
 * on that board. Best case by construction (pure copy, all cores); a
 * read-modify-write kernel cannot reach it.
 *
 *   gcc -O3 -march=native -fopenmp -o cpu_bandwidth tools/cpu_bandwidth.c
 *   OMP_NUM_THREADS=<all> ./cpu_bandwidth
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e3 + t.tv_nsec/1e6; }

int main(int argc, char **argv)
{
    size_t mb = argc > 1 ? (size_t)atoi(argv[1]) : 256;
    size_t n = mb << 20;
    char *a = malloc(n), *b = malloc(n);
    if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 1; }
    memset(a, 1, n); memset(b, 2, n);          /* fault the pages in */

    double best = 0;
    for (int rep = 0; rep < 10; rep++) {
        double t0 = now_ms();
#pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)(n >> 12); i++)
            memcpy(b + ((size_t)i << 12), a + ((size_t)i << 12), 4096);
        double ms = now_ms() - t0;
        double gbs = 2.0 * n / (ms / 1e3) / 1e9;   /* read + write */
        if (gbs > best) best = gbs;
    }
    int th = 1;
#ifdef _OPENMP
#pragma omp parallel
    { if (omp_get_thread_num() == 0) th = omp_get_num_threads(); }
#endif
    printf("achievable copy bandwidth: %.1f GB/s  (%zu MB buffers, %d threads, best of 10)\n",
           best, mb, th);
    return 0;
}
