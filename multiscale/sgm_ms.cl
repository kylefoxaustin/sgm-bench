/* multiscale/sgm_ms.cl — Configuration B for Mali (OpenCL 3.0).
 *
 * Same work-group shape as mali_cl/sgm.cl (16 work-items x uchar4 = D=64), with
 * Configuration B's additions: SobelX prefilter, census on the filtered image,
 * MATERIALISED cost volumes so the two scales can be averaged, eight paths with
 * a per-launch P1 (direction-weighted by the host), P2=200 with saturating
 * arithmetic throughout, and diagonal sweeps marching (x,y) together.
 */
#define D_DISP 64
#define CINVAL 63
#define P2V 200
#define DPI 4
#define WI  (D_DISP / DPI)

int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

__kernel void sobelx(__global const uchar *im, __global uchar *out, int W, int H)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    int xm = clampi(x-1,0,W-1), xp = clampi(x+1,0,W-1);
    int ym = clampi(y-1,0,H-1), yp = clampi(y+1,0,H-1);
    int gx = -im[ym*W+xm]+im[ym*W+xp]-2*im[y*W+xm]+2*im[y*W+xp]-im[yp*W+xm]+im[yp*W+xp];
    out[y*W+x] = (uchar)clampi(128 + (gx>>3), 0, 255);
}
__kernel void halfscale(__global const uchar *im, __global uchar *out, int W, int H)
{
    int w = W/2, h = H/2, x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    out[y*w+x] = (uchar)((im[(2*y)*W+2*x]+im[(2*y)*W+2*x+1]
                         +im[(2*y+1)*W+2*x]+im[(2*y+1)*W+2*x+1]+2)>>2);
}
__kernel void census(__global const uchar *im, __global ulong *out, int W, int H)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    uchar c = im[y*W+x]; ulong s = 0; int n = 0;
    for (int dy=-3; dy<=3; dy++) { int yy = clampi(y+dy,0,H-1);
        for (int dx=-4; dx<=4; dx++) { if (!dx && !dy) continue;
            if (im[yy*W+clampi(x+dx,0,W-1)] < c) s |= (ulong)1 << n;
            n++; } }
    out[y*W+x] = s;
}
__kernel void cost_build(__global const ulong *cl_, __global const ulong *cr_,
                         __global uchar *C, int W, int H)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    ulong a = cl_[y*W+x];
    __global uchar *c = C + ((size_t)y*W + x) * D_DISP;
    for (int d = 0; d < D_DISP; d++) {
        int xr = x - d;
        c[d] = (xr < 0) ? (uchar)CINVAL : (uchar)popcount(a ^ cr_[y*W+xr]);
    }
}
__kernel void cost_avg(__global uchar *C2, __global const uchar *C1, int W, int H)
{
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    __global uchar *c2 = C2 + ((size_t)y*W + x) * D_DISP;
    __global const uchar *c1 = C1 + ((size_t)(y/2)*(W/2) + (x/2)) * D_DISP;
    for (int d = 0; d < D_DISP; d++) c2[d] = (uchar)((c2[d] + c1[d/2] + 1) >> 1);
}
uchar4 load4(__global const uchar *c, int d0)
{ return (uchar4)(c[d0], c[d0+1], c[d0+2], c[d0+3]); }

uchar4 step4(uchar4 prev, uchar4 c, int lid, int p1,
             __local uchar *lbuf, __local uint *mbuf)
{
    lbuf[0] = 255; lbuf[D_DISP + 1] = 255;
    int b = 1 + lid * DPI;
    lbuf[b+0]=prev.s0; lbuf[b+1]=prev.s1; lbuf[b+2]=prev.s2; lbuf[b+3]=prev.s3;
    uint m = min(min((uint)prev.s0,(uint)prev.s1),min((uint)prev.s2,(uint)prev.s3));
    mbuf[lid] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WI >> 1; s > 0; s >>= 1) {
        if (lid < s) mbuf[lid] = min(mbuf[lid], mbuf[lid+s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    uint mn = mbuf[0], e = min(mn + P2V, 255u);
    uchar4 o;
    #define STEP(I)                                                       \
        { uint pv = lbuf[b+(I)];                                          \
          uint lo = min((uint)lbuf[b+(I)-1] + (uint)p1, 255u);            \
          uint hi = min((uint)lbuf[b+(I)+1] + (uint)p1, 255u);            \
          uint mm = min(min(pv,lo),min(hi,e));                            \
          o.s##I = (uchar)min((uint)c.s##I + (mm - mn), 255u); }
    STEP(0) STEP(1) STEP(2) STEP(3)
    #undef STEP
    barrier(CLK_LOCAL_MEM_FENCE);
    return o;
}
/* one work-group per LINE; the line is (sx,sy) with start derived from the
 * group id exactly as the CUDA version derives it */
__kernel __attribute__((reqd_work_group_size(WI,1,1)))
void agg_line(__global const uchar *C, __global ushort *S,
              int W, int H, int sx, int sy, int p1, int nlines)
{
    int line = get_group_id(0), lid = get_local_id(0);
    if (line >= nlines) return;
    int x0 = sx>0 ? 0 : (sx<0 ? W-1 : 0);
    int y0 = sy>0 ? 0 : (sy<0 ? H-1 : 0);
    int x, y;
    if (sy == 0)      { x = x0;  y = line; }
    else if (sx == 0) { x = line; y = y0; }
    else if (line < H){ x = x0;  y = line; }
    else              { x = x0 + sx*(line-H+1); y = y0; }
    __local uchar lbuf[D_DISP + 2];
    __local uint  mbuf[WI];
    int d0 = lid * DPI;
    uchar4 L = (uchar4)(0);
    int first = 1;
    while (x >= 0 && x < W && y >= 0 && y < H) {
        __global const uchar *c = C + ((size_t)y*W + x) * D_DISP;
        uchar4 cc = load4(c, d0);
        if (first) { L = cc; first = 0; }
        else L = step4(L, cc, lid, p1, lbuf, mbuf);
        __global ushort *s = S + ((size_t)y*W + x) * D_DISP + d0;
        s[0]+=L.s0; s[1]+=L.s1; s[2]+=L.s2; s[3]+=L.s3;
        x += sx; y += sy;
    }
}
__kernel void argmin_half(__global const ushort *S, __global uchar *disp, int W, int H)
{
    int ow = W/2, oh = H/2, x = get_global_id(0), y = get_global_id(1);
    if (x >= ow || y >= oh) return;
    __global const ushort *s = S + ((size_t)(2*y)*W + (2*x)) * D_DISP;
    ushort bv = s[0]; int bd = 0;
    for (int d = 1; d < D_DISP; d++) if (s[d] < bv) { bv = s[d]; bd = d; }
    disp[(size_t)y*ow + x] = (uchar)bd;
}
