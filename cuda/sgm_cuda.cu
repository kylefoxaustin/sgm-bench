/* sgm_cuda.cu — SGM on NVIDIA GPUs (Jetson Orin, Jetson Thor, discrete).
 *
 * A DELIBERATE PORT OF mali_cl/sgm.cl, NOT A REWRITE. That kernel is already
 * verified bit-exact against the golden on two Mali parts, so porting it
 * one-for-one carries the semantics across instead of re-deriving them and
 * re-discovering the same tie-break and edge-replication traps. The mapping is
 * mechanical:
 *
 *     __kernel            -> __global__          __local  -> __shared__
 *     get_group_id(0)     -> blockIdx.x          barrier  -> __syncthreads
 *     get_local_id(0)     -> threadIdx.x         popcount -> __popcll
 *     uchar4 .s0.s1.s2.s3 -> uchar4 .x.y.z.w
 *
 * Shape, unchanged: a block is 16 threads and each thread owns FOUR disparities
 * as a uchar4, so one block covers D=64. Neighbour exchange for the d-1 / d+1
 * terms goes through shared memory rather than warp shuffles -- same reasoning
 * as the OpenCL version, that correctness must not depend on a subgroup/warp
 * size, and here additionally because 16 threads is a HALF warp, so a shuffle
 * would need care that a __shared__ array does not.
 *
 * ⚠️ 16-thread blocks are deliberately poor CUDA occupancy. This is a
 * bit-exactness-first port; the number it produces is a floor for these parts,
 * exactly as the Mali number is a floor for the G720. Said plainly here so
 * nobody reads the result as what the hardware can do.
 */
extern "C" {
#include "sgm.h"
}
#include <cuda_runtime.h>
#include <stdio.h>

#define DPI 4
#define WI  (SGM_D / DPI)
#define CINVAL SGM_CENSUS_BITS

#define CK(v) do { cudaError_t e_ = (v); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    return -1; } } while (0)

__device__ __forceinline__ int clampi(int v, int lo, int hi)
{ return v < lo ? lo : (v > hi ? hi : v); }

/* ---- census: one thread per pixel. Bit n set iff neighbour < centre, n over
 *      dy outer / dx inner, centre skipped, edges replicated. ---- */
__global__ void k_census(const unsigned char *im, unsigned long long *out, int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const int rx = SGM_CENSUS_W / 2, ry = SGM_CENSUS_H / 2;
    const unsigned char c = im[y * W + x];
    unsigned long long s = 0; int n = 0;
    for (int dy = -ry; dy <= ry; dy++) {
        const int yy = clampi(y + dy, 0, H - 1);
        for (int dx = -rx; dx <= rx; dx++) {
            if (dx == 0 && dy == 0) continue;
            const int xx = clampi(x + dx, 0, W - 1);
            if (im[yy * W + xx] < c) s |= 1ULL << n;
            n++;
        }
    }
    out[(size_t)y * W + x] = s;
}

__device__ __forceinline__ uchar4 cost4(const unsigned long long *rl,
                                        const unsigned long long *rr, int x, int d0)
{
    const unsigned long long a = rl[x];
    uchar4 c; int xr;
    xr = x - (d0 + 0); c.x = (xr < 0) ? CINVAL : (unsigned char)__popcll(a ^ rr[xr]);
    xr = x - (d0 + 1); c.y = (xr < 0) ? CINVAL : (unsigned char)__popcll(a ^ rr[xr]);
    xr = x - (d0 + 2); c.z = (xr < 0) ? CINVAL : (unsigned char)__popcll(a ^ rr[xr]);
    xr = x - (d0 + 3); c.w = (xr < 0) ? CINVAL : (unsigned char)__popcll(a ^ rr[xr]);
    return c;
}

/* One recurrence step. lbuf is padded by one at each end so the d-1 / d+1 reads
 * never branch; the out-of-range guards are 255 and must survive the +P1 as 255,
 * which is why every add is clamped rather than allowed to wrap. */
__device__ __forceinline__ uchar4 step4(uchar4 prev, uchar4 c, int lid,
                                        unsigned char *lbuf, unsigned int *mbuf)
{
    lbuf[0] = 255; lbuf[SGM_D + 1] = 255;
    const int b = 1 + lid * DPI;
    lbuf[b + 0] = prev.x; lbuf[b + 1] = prev.y;
    lbuf[b + 2] = prev.z; lbuf[b + 3] = prev.w;

    unsigned int m = min(min((unsigned int)prev.x, (unsigned int)prev.y),
                         min((unsigned int)prev.z, (unsigned int)prev.w));
    mbuf[lid] = m;
    __syncthreads();
    for (int s = WI >> 1; s > 0; s >>= 1) {
        if (lid < s) mbuf[lid] = min(mbuf[lid], mbuf[lid + s]);
        __syncthreads();
    }
    const unsigned int mn = mbuf[0];
    const unsigned int e  = min(mn + SGM_P2, 255u);

    uchar4 o;
#define STEP(I, F)                                                        \
    { unsigned int pv = lbuf[b + (I)];                                    \
      unsigned int lo = min((unsigned int)lbuf[b + (I) - 1] + SGM_P1, 255u); \
      unsigned int hi = min((unsigned int)lbuf[b + (I) + 1] + SGM_P1, 255u); \
      unsigned int mm = min(min(pv, lo), min(hi, e));                     \
      o.F = (unsigned char)min((unsigned int)c.F + (mm - mn), 255u); }
    STEP(0, x) STEP(1, y) STEP(2, z) STEP(3, w)
#undef STEP
    __syncthreads();
    return o;
}

__global__ void k_agg_horiz(const unsigned long long *cl_, const unsigned long long *cr_,
                            unsigned short *S, int W, int H, int dir)
{
    const int y = blockIdx.x, lid = threadIdx.x;
    if (y >= H) return;
    const unsigned long long *rl = cl_ + (size_t)y * W, *rr = cr_ + (size_t)y * W;
    __shared__ unsigned char lbuf[SGM_D + 2];
    __shared__ unsigned int  mbuf[WI];
    const int d0 = lid * DPI;
    uchar4 L = make_uchar4(0, 0, 0, 0);
    const int xs = (dir > 0) ? 0 : W - 1;
    for (int i = 0; i < W; i++) {
        const int x = xs + dir * i;
        uchar4 c = cost4(rl, rr, x, d0);
        L = (i == 0) ? c : step4(L, c, lid, lbuf, mbuf);
        unsigned short *s = S + ((size_t)y * W + x) * SGM_D + d0;
        s[0] += L.x; s[1] += L.y; s[2] += L.z; s[3] += L.w;
    }
}

__global__ void k_agg_vert(const unsigned long long *cl_, const unsigned long long *cr_,
                           unsigned short *S, int W, int H, int dir)
{
    const int x = blockIdx.x, lid = threadIdx.x;
    if (x >= W) return;
    __shared__ unsigned char lbuf[SGM_D + 2];
    __shared__ unsigned int  mbuf[WI];
    const int d0 = lid * DPI;
    uchar4 L = make_uchar4(0, 0, 0, 0);
    const int ys = (dir > 0) ? 0 : H - 1;
    for (int i = 0; i < H; i++) {
        const int y = ys + dir * i;
        uchar4 c = cost4(cl_ + (size_t)y * W, cr_ + (size_t)y * W, x, d0);
        L = (i == 0) ? c : step4(L, c, lid, lbuf, mbuf);
        unsigned short *s = S + ((size_t)y * W + x) * SGM_D + d0;
        s[0] += L.x; s[1] += L.y; s[2] += L.z; s[3] += L.w;
    }
}

/* argmin: lowest d wins ties -- the same arbitrary rule as every other target,
 * because a different one is still correct SGM and a different hash. */
__global__ void k_argmin(const unsigned short *S, unsigned char *disp, int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const unsigned short *s = S + ((size_t)y * W + x) * SGM_D;
    unsigned short bv = s[0]; int bd = 0;
    for (int d = 1; d < SGM_D; d++) if (s[d] < bv) { bv = s[d]; bd = d; }
    disp[(size_t)y * W + x] = (unsigned char)bd;
}

static unsigned char *d_l, *d_r, *d_disp;
static unsigned long long *d_cl, *d_cr;
static unsigned short *d_S;
static int initW, initH, ready;

static int setup(int W, int H)
{
    const size_t N = (size_t)W * H;
    cudaDeviceProp p; int dev = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&p, dev));
    fprintf(stderr, "# cuda device: %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor,
            p.multiProcessorCount);
    CK(cudaMalloc(&d_l, N));            CK(cudaMalloc(&d_r, N));
    CK(cudaMalloc(&d_cl, N * 8));       CK(cudaMalloc(&d_cr, N * 8));
    CK(cudaMalloc(&d_S, N * SGM_D * sizeof(unsigned short)));
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

    double t0 = now_ms();
    CK(cudaMemcpy(d_l, left,  N, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_r, right, N, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_S, 0, N * SGM_D * sizeof(unsigned short)));

    dim3 cb(16, 4), cg((W + 15) / 16, (H + 3) / 4);
    k_census<<<cg, cb>>>(d_l, d_cl, W, H);
    k_census<<<cg, cb>>>(d_r, d_cr, W, H);
    CK(cudaDeviceSynchronize());
    double t1 = now_ms();

    for (int i = 0; i < 4 && i < SGM_PATHS; i++) {
        if (i < 2) k_agg_horiz<<<H, WI>>>(d_cl, d_cr, d_S, W, H, i ? -1 : +1);
        else       k_agg_vert <<<W, WI>>>(d_cl, d_cr, d_S, W, H, (i == 2) ? +1 : -1);
        CK(cudaDeviceSynchronize());
    }
    double t2 = now_ms();

    k_argmin<<<cg, cb>>>(d_S, d_disp, W, H);
    CK(cudaMemcpy(disp, d_disp, N, cudaMemcpyDeviceToHost));
    CK(cudaDeviceSynchronize());
    double t3 = now_ms();

    if (t) { t->census_ms = t1 - t0; t->cost_ms = -1;
             t->aggregate_ms = t2 - t1; t->argmin_ms = t3 - t2; }
    return 0;
}

extern "C" const sgm_impl SGM_IMPL = { "cuda", cuda_run };
