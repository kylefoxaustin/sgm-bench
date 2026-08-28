/* ============================================================================
 * sgm_census_cost.c
 *
 * Companion to sgm_aggregate_neon.c: benchmarks the two SGM pipeline stages
 * that PRECEDE cost aggregation.
 *
 *   (a) Census transform - 9x7 local binary descriptor, 63 bits packed into a
 *       uint64 per pixel. Each output depends only on a local window, so this
 *       stage is embarrassingly parallel with no recurrence.
 *
 *   (b) Hamming matching cost - for each pixel and each of D=64 candidate
 *       disparities, the Hamming distance between the left descriptor and the
 *       shifted right descriptor. Reduces to XOR + population count. Also fully
 *       parallel.
 *
 * Both stages are threaded with OpenMP over image rows. Neither is hand
 * vectorised: they rely on -O3 autovectorisation and the hardware population
 * count instruction. They are included because, once aggregation is vectorised,
 * these two stages become the larger share of total runtime and are therefore
 * the next optimisation target.
 *
 * Measured on 6 x Cortex-A55 @ 1.8 GHz, gcc 13.3, -O3 -march=native -fopenmp:
 *
 *   960x400    1 thread    census (2 images) 103.5 ms   Hamming cost 164.0 ms
 *   960x400    6 threads   census (2 images)  18.0 ms   Hamming cost  28.4 ms
 *   1920x800   6 threads   census (2 images)  71.7 ms   Hamming cost 113.1 ms
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -o sgm_census_cost sgm_census_cost.c -lm
 * Run:
 *   ./sgm_census_cost [W] [H] [threads]        (default 960 400, 0 = OMP default)
 *
 * Inputs are a fixed pseudo-random pattern (seed 1); a sampled checksum is
 * printed so results can be compared across builds and thread counts.
 * ============================================================================
 */

/* Census (9x7) + Hamming data cost, NEON vs scalar. Both stages are
   embarrassingly parallel (no recurrence) so they should vectorise cleanly.
   Measures the two pipeline stages that aggregation does NOT cover. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__ARM_NEON)||defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#endif
#define D 64
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec+1e-9*t.tv_nsec;}

/* 9x7 census -> 63 bits in a uint64 */
static void census_scalar(const uint8_t*im,uint64_t*out,int W,int H){
#pragma omp parallel for schedule(static)
  for(int y=3;y<H-3;y++)for(int x=4;x<W-4;x++){
    uint8_t p=im[y*W+x]; uint64_t s=0; int n=0;
    for(int dy=-3;dy<=3;dy++)for(int dx=-4;dx<=4;dx++){
      if(!dx&&!dy)continue;
      s|=(uint64_t)(im[(y+dy)*W+(x+dx)]<p)<<n; n++;}
    out[y*W+x]=s;
  }
}
/* Hamming cost over D disparities; popcount is the hot op */
static void dcost_scalar(const uint64_t*l,const uint64_t*r,uint8_t*C,int W,int H){
#pragma omp parallel for schedule(static)
  for(int y=3;y<H-3;y++)for(int x=4;x<W-4;x++){
    uint64_t a=l[y*W+x]; uint8_t*c=C+((size_t)y*W+x)*D;
    for(int d=0;d<D;d++){int xr=x-d;
      c[d]=(xr<4)?63:(uint8_t)__builtin_popcountll(a^r[y*W+xr]);}
  }
}

int main(int argc,char**argv){
  int W=argc>1?atoi(argv[1]):960,H=argc>2?atoi(argv[2]):400,th=argc>3?atoi(argv[3]):0;
#ifdef _OPENMP
  if(th>0) omp_set_num_threads(th);
#endif
  uint8_t*im=malloc((size_t)W*H),*im2=malloc((size_t)W*H);
  uint64_t*cl=calloc((size_t)W*H,8),*cr=calloc((size_t)W*H,8);
  uint8_t*C=malloc((size_t)W*H*D);
  if(!C){printf("alloc fail\n");return 1;}
  srand(1);
  for(size_t i=0;i<(size_t)W*H;i++){im[i]=rand();im2[i]=rand();}
  int nt=1;
#ifdef _OPENMP
#pragma omp parallel
  {if(omp_get_thread_num()==0)nt=omp_get_num_threads();}
#endif
  double t0=now(); census_scalar(im,cl,W,H); census_scalar(im2,cr,W,H);
  double t1=now(); dcost_scalar(cl,cr,C,W,H); double t2=now();
  uint64_t chk=0; for(size_t i=0;i<(size_t)W*H*D;i+=997) chk+=C[i];
  printf("%dx%d th=%-2d  census(2x) %7.1f ms   dcost %7.1f ms   chk=%llu\n",
    W,H,nt,(t1-t0)*1e3,(t2-t1)*1e3,(unsigned long long)chk);
  return 0;
}
