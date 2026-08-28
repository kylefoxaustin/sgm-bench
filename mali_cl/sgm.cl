/* sgm.cl — SGM for Mali Valhall (G310 / G720-Immortalis), OpenCL 3.0.
 *
 * Shape, following the spec: a work-group is 16 work-items and each work-item
 * owns FOUR disparities as a uchar4, so one work-group covers D=64 and matches
 * the measured subgroup size of 16.
 *
 * NEIGHBOUR EXCHANGE VIA LOCAL MEMORY, NOT SUBGROUP SHUFFLE. The spec suggests
 * sub_group_shuffle for the d-1 / d+1 terms. Local memory is used instead
 * because it is correct for ANY subgroup size, and the subgroup size is exactly
 * the thing that was misreported during bring-up (a wrong enum said "1"; the
 * kernel-level query said 16). A kernel whose correctness depends on a number
 * the driver reports inconsistently is a kernel that will fail somewhere else.
 * The exchange is 2 loads per pixel per direction and is not the bottleneck.
 *
 * Bit-exactness: census here MUST reproduce the CPU bit order exactly --
 * bit n set iff neighbour < centre, n incrementing dy-outer/dx-inner, centre
 * skipped, edge-replicated. Any deviation changes every Hamming distance.
 */

#ifndef D_DISP
#define D_DISP 64
#endif
#ifndef CW
#define CW 9
#endif
#ifndef CH
#define CH 7
#endif
#ifndef P1V
#define P1V 20
#endif
#ifndef P2V
#define P2V 192
#endif
#define CBITS (CW * CH - 1)
#define CINVAL CBITS
#define DPI 4                   /* disparities per work-item */
#define WI  (D_DISP / DPI)      /* work-items per group = 16 at D=64 */

int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ---- census: one work-item per pixel ---- */
__kernel void census(__global const uchar *im, __global ulong *out,
                     const int W, const int H)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    const int rx = CW / 2, ry = CH / 2;
    const uchar c = im[y * W + x];
    ulong s = 0; int n = 0;
    for (int dy = -ry; dy <= ry; dy++) {
        const int yy = clampi(y + dy, 0, H - 1);
        for (int dx = -rx; dx <= rx; dx++) {
            if (dx == 0 && dy == 0) continue;
            const int xx = clampi(x + dx, 0, W - 1);
            if (im[yy * W + xx] < c) s |= (ulong)1 << n;
            n++;
        }
    }
    out[y * W + x] = s;
}

/* Hamming cost for this work-item's four disparities at column x of a row. */
static uchar4 cost4(__global const ulong *rl, __global const ulong *rr,
                    int x, int d0)
{
    const ulong a = rl[x];
    uchar4 c;
    int xr;
    xr = x - (d0 + 0); c.s0 = (xr < 0) ? CINVAL : (uchar)popcount(a ^ rr[xr]);
    xr = x - (d0 + 1); c.s1 = (xr < 0) ? CINVAL : (uchar)popcount(a ^ rr[xr]);
    xr = x - (d0 + 2); c.s2 = (xr < 0) ? CINVAL : (uchar)popcount(a ^ rr[xr]);
    xr = x - (d0 + 3); c.s3 = (xr < 0) ? CINVAL : (uchar)popcount(a ^ rr[xr]);
    return c;
}

/* One recurrence step, shared by both sweep kernels.
 * lbuf is __local uchar[D_DISP + 2] with a 1-element guard at each end so the
 * d-1 / d+1 reads never branch; mbuf is __local uint[WI] for the min reduction.
 * Returns the new L for this work-item's four disparities. */
static uchar4 step4(uchar4 prev, uchar4 c, int lid,
                    __local uchar *lbuf, __local uint *mbuf)
{
    /* publish prev into the padded buffer */
    lbuf[0] = 255; lbuf[D_DISP + 1] = 255;
    const int b = 1 + lid * DPI;
    lbuf[b + 0] = prev.s0; lbuf[b + 1] = prev.s1;
    lbuf[b + 2] = prev.s2; lbuf[b + 3] = prev.s3;

    /* min over all D: per-item min then a tree over WI items */
    uint m = min(min((uint)prev.s0, (uint)prev.s1),
                 min((uint)prev.s2, (uint)prev.s3));
    mbuf[lid] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = WI >> 1; s > 0; s >>= 1) {
        if (lid < s) mbuf[lid] = min(mbuf[lid], mbuf[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const uint mn = mbuf[0];
    const uint e  = min(mn + P2V, 255u);

    uchar4 o;
    /* saturating adds: an out-of-range neighbour is 255 and must stay 255 */
    #define STEP(I)                                                        \
        { uint pv = lbuf[b + (I)];                                         \
          uint lo = min((uint)lbuf[b + (I) - 1] + P1V, 255u);               \
          uint hi = min((uint)lbuf[b + (I) + 1] + P1V, 255u);               \
          uint mm = min(min(pv, lo), min(hi, e));                          \
          o.s##I = (uchar)min((uint)c.s##I + (mm - mn), 255u); }
    STEP(0) STEP(1) STEP(2) STEP(3)
    #undef STEP
    barrier(CLK_LOCAL_MEM_FENCE);
    return o;
}

/* ---- horizontal sweep: one work-group per ROW, marches along x.
 *      dir = +1 for the L path (left->right), -1 for R. ---- */
__kernel __attribute__((reqd_work_group_size(WI, 1, 1)))
void agg_horiz(__global const ulong *cl_, __global const ulong *cr_,
               __global ushort *S, const int W, const int H, const int dir)
{
    const int y   = get_group_id(0);
    const int lid = get_local_id(0);
    if (y >= H) return;
    __global const ulong *rl = cl_ + (size_t)y * W, *rr = cr_ + (size_t)y * W;
    __local uchar lbuf[D_DISP + 2];
    __local uint  mbuf[WI];
    const int d0 = lid * DPI;

    uchar4 L = (uchar4)(0);
    const int xs = (dir > 0) ? 0 : W - 1;
    for (int i = 0; i < W; i++) {
        const int x = xs + dir * i;
        uchar4 c = cost4(rl, rr, x, d0);
        L = (i == 0) ? c : step4(L, c, lid, lbuf, mbuf);
        __global ushort *s = S + ((size_t)y * W + x) * D_DISP + d0;
        s[0] += L.s0; s[1] += L.s1; s[2] += L.s2; s[3] += L.s3;
    }
}

/* ---- vertical sweep: one work-group per COLUMN, marches along y.
 *      dir = +1 for U (top->bottom), -1 for D. ---- */
__kernel __attribute__((reqd_work_group_size(WI, 1, 1)))
void agg_vert(__global const ulong *cl_, __global const ulong *cr_,
              __global ushort *S, const int W, const int H, const int dir)
{
    const int x   = get_group_id(0);
    const int lid = get_local_id(0);
    if (x >= W) return;
    __local uchar lbuf[D_DISP + 2];
    __local uint  mbuf[WI];
    const int d0 = lid * DPI;

    uchar4 L = (uchar4)(0);
    const int ys = (dir > 0) ? 0 : H - 1;
    for (int i = 0; i < H; i++) {
        const int y = ys + dir * i;
        uchar4 c = cost4(cl_ + (size_t)y * W, cr_ + (size_t)y * W, x, d0);
        L = (i == 0) ? c : step4(L, c, lid, lbuf, mbuf);
        __global ushort *s = S + ((size_t)y * W + x) * D_DISP + d0;
        s[0] += L.s0; s[1] += L.s1; s[2] += L.s2; s[3] += L.s3;
    }
}

/* ---- argmin: one work-item per pixel, lowest d wins ties ---- */
__kernel void argmin_k(__global const ushort *S, __global uchar *disp,
                       const int W, const int H)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if (x >= W || y >= H) return;
    __global const ushort *s = S + ((size_t)y * W + x) * D_DISP;
    ushort bv = s[0]; int bd = 0;
    for (int d = 1; d < D_DISP; d++) if (s[d] < bv) { bv = s[d]; bd = d; }
    disp[(size_t)y * W + x] = (uchar)bd;
}
