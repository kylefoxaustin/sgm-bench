/* sgm_cuda_opt.cu — SGM for NVIDIA, written for CUDA instead of transliterated.
 *
 * cuda/sgm_cuda.cu is a one-for-one port of the OpenCL kernel. It was the right
 * way to REACH bit-exactness -- it hit the golden first try on two parts -- and
 * the wrong shape for this machine. What it inherited from OpenCL:
 *
 *   16-thread blocks              half a warp; 16 of 32 lanes idle, always
 *   __syncthreads() PER PIXEL     three of them, inside the recurrence
 *   L and the reduction in shared memory
 *   1080 blocks x 16 threads      17k threads to fill a 20-SM GPU
 *
 * This version keeps the algorithm and the hash and changes the shape:
 *
 *   ONE WARP OWNS ONE SCANLINE'S ENTIRE DISPARITY RANGE. D=64 over 32 lanes is
 *   DPT=2 disparities per lane, held in REGISTERS. D=128 gives DPT=4; the code
 *   is written for any D that is a multiple of 32.
 *
 *   THE MIN OVER d IS A WARP SHUFFLE, not a shared-memory tree: 5 steps of
 *   __shfl_xor_sync with no barrier at all. The OpenCL version needed 4 barriers
 *   and a __shared__ array for the same reduction, per pixel, per path.
 *
 *   THE d-1 / d+1 EXCHANGE IS ALSO A SHUFFLE. Each lane owns a contiguous run
 *   of disparities, so only the run's two ends cross a lane boundary: the low
 *   end needs lane-1's LAST slot, the high end needs lane+1's FIRST slot. Two
 *   shuffles per step; every interior neighbour is a register already in hand.
 *
 *   NO __syncthreads ANYWHERE IN THE RECURRENCE. Warps are independent
 *   scanlines, so there is nothing to synchronise. 8 warps per block is purely
 *   an occupancy choice, not a communication one.
 *
 * ⚠️ WARP-SYNCHRONOUS CODE NEEDS EXPLICIT MASKS. Every shuffle passes 0xffffffff
 * and every warp is fully populated by construction (blockDim.x == 32). This is
 * the class of code where "it worked on my GPU" hides a latent divergence bug,
 * which is exactly why the golden hash is checked in the same run as the timing.
 *
 * Ties still go to the lowest d, out-of-range neighbours are still 255, and the
 * adds still clamp rather than wrap. Those are what make the hash reproduce; the
 * shape is free to change underneath them.
 */
extern "C" {
#include "sgm.h"
}
#include <cuda_runtime.h>
#include <stdio.h>

#define WARP 32
#define DPT  (SGM_D / WARP)          /* disparities per lane: 2 at D=64 */
#ifndef WPB
#define WPB  4                       /* warps per block = scanlines in flight */
#endif
#define CINVAL SGM_CENSUS_BITS
#define FULL 0xffffffffu

#if (SGM_D % WARP) != 0
#error "sgm_cuda_opt requires SGM_D to be a multiple of 32"
#endif
#if SGM_PATHS != 4
/* This kernel implements FOUR paths. SGM_PATHS=8 used to compile silently and
 * compute four anyway -- the diagonal kernels do not exist -- so a build asking
 * for eight got a correct-looking four-path answer with an eight-path label.
 * Regenerating the golden would have caught it, but this repo argues elsewhere
 * that a compile-time refusal beats a silently wrong result, and did not apply
 * that to itself here. */
#error "sgm_cuda_opt implements 4 paths only; SGM_PATHS=8 has no diagonal kernels"
#endif

#define CK(v) do { cudaError_t e_ = (v); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    return -1; } } while (0)

__device__ __forceinline__ int clampi(int v, int lo, int hi)
{ return v < lo ? lo : (v > hi ? hi : v); }

/* ---- census: unchanged in substance, but one warp per 32 consecutive x so the
 *      image loads coalesce. ---- */
__global__ void ko_census(const unsigned char *__restrict__ im,
                          unsigned long long *__restrict__ out, int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int rx = SGM_CENSUS_W / 2, ry = SGM_CENSUS_H / 2;
    const unsigned char c = im[(size_t)y * W + x];
    unsigned long long s = 0; int n = 0;
    for (int dy = -ry; dy <= ry; dy++) {
        const int yy = clampi(y + dy, 0, H - 1);
        const unsigned char *row = im + (size_t)yy * W;
        for (int dx = -rx; dx <= rx; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (row[clampi(x + dx, 0, W - 1)] < c) s |= 1ULL << n;
            n++;
        }
    }
    out[(size_t)y * W + x] = s;
}

/* One recurrence step for this lane's DPT disparities, entirely in registers.
 * prev/cur are the lane's slice of L; `mn` is min over ALL D (warp-reduced). */
__device__ __forceinline__ void step_warp(const unsigned char *prev, const unsigned char *c,
                                          unsigned char *cur, unsigned int mn, int lane)
{
    /* the two neighbours that live in adjacent lanes */
    unsigned int loNb = __shfl_up_sync(FULL, (unsigned int)prev[DPT - 1], 1);
    unsigned int hiNb = __shfl_down_sync(FULL, (unsigned int)prev[0], 1);
    if (lane == 0)        loNb = 255;      /* d-1 below 0 does not exist */
    if (lane == WARP - 1) hiNb = 255;      /* d+1 above D-1 does not exist */

    const unsigned int e = min(mn + SGM_P2, 255u);
    for (int k = 0; k < DPT; k++) {
        unsigned int lo = (k == 0)       ? loNb : (unsigned int)prev[k - 1];
        unsigned int hi = (k == DPT - 1) ? hiNb : (unsigned int)prev[k + 1];
        unsigned int m = prev[k];
        m = min(m, min(lo + SGM_P1, 255u));
        m = min(m, min(hi + SGM_P1, 255u));
        m = min(m, e);
        cur[k] = (unsigned char)min((unsigned int)c[k] + (m - mn), 255u);
    }
}

__device__ __forceinline__ unsigned int warp_min(unsigned int v)
{
    for (int o = WARP / 2; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(FULL, v, o));
    return v;
}

__device__ __forceinline__ void cost_lane(const unsigned long long *rl,
                                          const unsigned long long *rr,
                                          int x, int d0, unsigned char *c)
{
    const unsigned long long a = rl[x];
    for (int k = 0; k < DPT; k++) {
        const int xr = x - (d0 + k);
        c[k] = (xr < 0) ? (unsigned char)CINVAL
                        : (unsigned char)__popcll(a ^ rr[xr]);
    }
}


/* S / scratch update, and the reason there are three modes.
 *
 * The aggregate phase is MEMORY BOUND, not compute bound: measured on Thor it
 * moves ~2.0 GB in 11.4 ms = 175 GB/s against a 247 GB/s copy ceiling measured
 * on the same part. So the lever is TRAFFIC, and S dominates it -- 265 MB
 * touched seven times (one store plus three read-modify-writes) is 1.86 of
 * those 2.0 GB.
 *
 * Pairing the paths cuts S to four touches. Each pair runs its second direction
 * FIRST into a uint8 scratch (half the width of S, since one path's L fits in a
 * byte), then the first direction reads the scratch back and commits both at
 * once. 7 touches of S becomes 4 touches of S plus 2 of a half-width scratch:
 * 1.99 GB -> 1.46 GB.
 *
 *   SCRATCH  write this path's L to scratch, do not touch S
 *   STORE    read scratch, add this path's L, STORE to S   (first pair)
 *   ACCUM    read scratch, add this path's L, ADD into S   (second pair)
 *
 * STORE also removes the 265 MB memset that a pure-accumulate scheme needs.
 * At DPT=2 a lane's pair is one ushort2 / uchar2 access, so every one of these
 * is a single coalesced transaction per lane. */
#define M_SCRATCH 0
#define M_STORE   1
#define M_ACCUM   2

template <int MODE>
__device__ __forceinline__ void s_commit(unsigned short *s, unsigned char *sc,
                                         const unsigned char *L)
{
/* The vector width follows DPT. Measured fact worth keeping: when only DPT==2
 * had a vector path, generalising to DPT 1/2/4 changed NOTHING (10.59/13.19/
 * 21.72 ms, inside noise) -- nvcc already emits efficient accesses for the
 * contiguous scalar loop. The explicit paths are kept as clearer code, not as
 * an optimisation, and DPT=8 (D=256) still takes the #else loop for the same
 * reason: it costs nothing. */
#if DPT == 1
    if (MODE == M_SCRATCH) { *sc = L[0]; return; }
    unsigned short v = (unsigned short)(*sc + L[0]);
    *s = (MODE == M_ACCUM) ? (unsigned short)(*s + v) : v;
#elif DPT == 2
    if (MODE == M_SCRATCH) { uchar2 v; v.x = L[0]; v.y = L[1]; *(uchar2 *)sc = v; return; }
    uchar2 r = *(const uchar2 *)sc;
    ushort2 v;
    v.x = (unsigned short)(r.x + L[0]);
    v.y = (unsigned short)(r.y + L[1]);
    if (MODE == M_ACCUM) { ushort2 o = *(const ushort2 *)s; v.x += o.x; v.y += o.y; }
    *(ushort2 *)s = v;
#elif DPT == 4
    if (MODE == M_SCRATCH) {
        uchar4 v; v.x = L[0]; v.y = L[1]; v.z = L[2]; v.w = L[3];
        *(uchar4 *)sc = v; return;
    }
    uchar4 r = *(const uchar4 *)sc;
    ushort4 v;
    v.x = (unsigned short)(r.x + L[0]); v.y = (unsigned short)(r.y + L[1]);
    v.z = (unsigned short)(r.z + L[2]); v.w = (unsigned short)(r.w + L[3]);
    if (MODE == M_ACCUM) {
        ushort4 o = *(const ushort4 *)s;
        v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
    }
    *(ushort4 *)s = v;
#else
    if (MODE == M_SCRATCH) { for (int k = 0; k < DPT; k++) sc[k] = L[k]; return; }
    for (int k = 0; k < DPT; k++) {
        unsigned short v = (unsigned short)(sc[k] + L[k]);
        s[k] = (MODE == M_ACCUM) ? (unsigned short)(s[k] + v) : v;
    }
#endif
}

/* Horizontal sweep: one WARP per row. dir=+1 is L, -1 is R. */
template <int MODE>
__global__ void ko_horiz(const unsigned long long *__restrict__ cl_,
                         const unsigned long long *__restrict__ cr_,
                         unsigned short *__restrict__ S, unsigned char *__restrict__ SC,
                         int W, int H, int dir)
{
    const int lane = threadIdx.x;
    const int y = blockIdx.x * WPB + threadIdx.y;
    if (y >= H) return;
    const unsigned long long *rl = cl_ + (size_t)y * W, *rr = cr_ + (size_t)y * W;
    const int d0 = lane * DPT;

    unsigned char L[DPT], c[DPT], nx[DPT];
    const int xs = (dir > 0) ? 0 : W - 1;
    for (int i = 0; i < W; i++) {
        const int x = xs + dir * i;
        cost_lane(rl, rr, x, d0, c);
        if (i == 0) { for (int k = 0; k < DPT; k++) L[k] = c[k]; }
        else {
            unsigned int mn = 255;
            for (int k = 0; k < DPT; k++) mn = min(mn, (unsigned int)L[k]);
            mn = warp_min(mn);
            step_warp(L, c, nx, mn, lane);
            for (int k = 0; k < DPT; k++) L[k] = nx[k];
        }
        { size_t o = ((size_t)y * W + x) * SGM_D + d0;
          s_commit<MODE>(S + o, SC + o, L); }
    }
}

/* Vertical sweep: one WARP per column. dir=+1 is U, -1 is D. */
template <int MODE>
__global__ void ko_vert(const unsigned long long *__restrict__ cl_,
                        const unsigned long long *__restrict__ cr_,
                        unsigned short *__restrict__ S, unsigned char *__restrict__ SC,
                        int W, int H, int dir)
{
    const int lane = threadIdx.x;
    const int x = blockIdx.x * WPB + threadIdx.y;
    if (x >= W) return;
    const int d0 = lane * DPT;

    unsigned char L[DPT], c[DPT], nx[DPT];
    const int ys = (dir > 0) ? 0 : H - 1;
    for (int i = 0; i < H; i++) {
        const int y = ys + dir * i;
        cost_lane(cl_ + (size_t)y * W, cr_ + (size_t)y * W, x, d0, c);
        if (i == 0) { for (int k = 0; k < DPT; k++) L[k] = c[k]; }
        else {
            unsigned int mn = 255;
            for (int k = 0; k < DPT; k++) mn = min(mn, (unsigned int)L[k]);
            mn = warp_min(mn);
            step_warp(L, c, nx, mn, lane);
            for (int k = 0; k < DPT; k++) L[k] = nx[k];
        }
        { size_t o = ((size_t)y * W + x) * SGM_D + d0;
          s_commit<MODE>(S + o, SC + o, L); }
    }
}

/* argmin: the naive port gave one thread the whole 64-iteration scan, and the
 * cost gate measured that phase at 3.9x the efficiency of census. Here a warp
 * cooperates on one pixel: each lane scans its DPT and a shuffle reduction picks
 * the winner, with the lowest d breaking ties as everywhere else. */
__global__ void ko_argmin(const unsigned short *__restrict__ S,
                          unsigned char *__restrict__ disp, int W, int H)
{
    const int lane = threadIdx.x;
    const size_t p = (size_t)blockIdx.x * WPB + threadIdx.y;
    if (p >= (size_t)W * H) return;
    const unsigned short *s = S + p * SGM_D + lane * DPT;

    unsigned int best = 0xffffu; int bd = 0;
    for (int k = 0; k < DPT; k++)
        if ((unsigned int)s[k] < best) { best = s[k]; bd = lane * DPT + k; }

    /* pack (cost, d) so one reduction settles both, lowest d winning ties */
    unsigned int key = (best << 16) | (unsigned int)bd;
    for (int o = WARP / 2; o > 0; o >>= 1) key = min(key, __shfl_xor_sync(FULL, key, o));
    if (lane == 0) disp[p] = (unsigned char)(key & 0xffffu);
}

static unsigned char *d_l, *d_r, *d_disp;
static unsigned long long *d_cl, *d_cr;
static unsigned short *d_S;
static unsigned char *d_SC;   /* half-width scratch: one path's L, uint8 */
static int initW, initH, ready;

static int setup(int W, int H)
{
    const size_t N = (size_t)W * H;
    cudaDeviceProp p; int dev = 0;
    CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&p, dev));
    fprintf(stderr, "# cuda device: %s (sm_%d%d, %d SMs) D=%d DPT=%d\n",
            p.name, p.major, p.minor, p.multiProcessorCount, SGM_D, DPT);
    CK(cudaMalloc(&d_l, N));       CK(cudaMalloc(&d_r, N));
    CK(cudaMalloc(&d_cl, N * 8));  CK(cudaMalloc(&d_cr, N * 8));
    CK(cudaMalloc(&d_S, N * SGM_D * sizeof(unsigned short)));
    CK(cudaMalloc(&d_SC, N * SGM_D));
    CK(cudaMalloc(&d_disp, N));
    initW = W; initH = H; ready = 1;
    return 0;
}

static int cuda_run(const uint8_t *left, const uint8_t *right, int W, int H,
                    uint8_t *disp, int threads, sgm_stage_times *t)
{
    (void)threads;
    if (!ready || W != initW || H != initH) { if (setup(W, H)) return -1; }
    const size_t N = (size_t)W * H;

    /* Transfers are timed SEPARATELY. They used to sit inside census_ms and
     * argmin_ms, which meant every per-phase figure derived from those was
     * really phase-plus-copy. */
    double tA = now_ms();
    CK(cudaMemcpy(d_l, left,  N, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_r, right, N, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    double t0 = now_ms();
    /* no memset: path 0 STORES into S rather than accumulating into zeros */

    dim3 cb(32, 8), cg((W + 31) / 32, (H + 7) / 8);
    ko_census<<<cg, cb>>>(d_l, d_cl, W, H);
    ko_census<<<cg, cb>>>(d_r, d_cr, W, H);
    CK(cudaDeviceSynchronize());
    double t1 = now_ms();

    dim3 wb(WARP, WPB);
    const int gh = (H + WPB - 1) / WPB, gw = (W + WPB - 1) / WPB;
    /* each pair: second direction into scratch, first direction commits both */
    ko_horiz<M_SCRATCH><<<gh, wb>>>(d_cl, d_cr, d_S, d_SC, W, H, -1);
    ko_horiz<M_STORE  ><<<gh, wb>>>(d_cl, d_cr, d_S, d_SC, W, H, +1);
    if (SGM_PATHS > 2) {
        ko_vert<M_SCRATCH><<<gw, wb>>>(d_cl, d_cr, d_S, d_SC, W, H, -1);
        ko_vert<M_ACCUM  ><<<gw, wb>>>(d_cl, d_cr, d_S, d_SC, W, H, +1);
    }
    CK(cudaDeviceSynchronize());
    double t2 = now_ms();

    ko_argmin<<<(unsigned)((N + WPB - 1) / WPB), wb>>>(d_S, d_disp, W, H);
    CK(cudaDeviceSynchronize());
    double t3 = now_ms();
    CK(cudaMemcpy(disp, d_disp, N, cudaMemcpyDeviceToHost));
    CK(cudaDeviceSynchronize());
    double tB = now_ms();

    if (t) { t->census_ms = t1 - t0; t->cost_ms = -1;
             t->aggregate_ms = t2 - t1; t->argmin_ms = t3 - t2;
             t->transfer_ms = (t0 - tA) + (tB - t3); t->threads_used = 0; }
    return 0;
}

extern "C" const sgm_impl SGM_IMPL = { "cuda_opt", cuda_run };
