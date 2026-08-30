/* sgmms_kernels.h — Configuration B (multiscale SGM) kernels for Hexagon v73.
 *
 * Semantics are sgm-bench/multiscale/sgm_ms_ref.c, reproduced EXACTLY:
 *   input WxH grey -> output (W/2)x(H/2) integer disparity, D=64
 *   two scales, both full 64-d SGM; the two DATA COSTS are averaged
 *     merged(x,y,d) = (c2[d] + c1_at(x/2,y/2)[d/2] + 1) >> 1
 *   census 9x7 on a horizontal-Sobel prefiltered image
 *   8 aggregation paths, per-direction P1 (40 H / 20 V / 10 diag), P2=200
 *   SATURATING adds everywhere (with P2=200 the bracket can exceed 255)
 *   cost 63 when x-d < 0 (not 62); ties to lowest d; no postprocessing.
 *
 * Included by BOTH sgmms_imp.c (ships) and sim/msb.c (validates), so what the
 * simulator checks is byte-for-byte what runs on the board.
 *
 * This file is Config-B specific: it requires SGM_D == 64 (DPAD=64, PPV=2,
 * VPC=1, no pad lanes).  P2=200 breaks two Config-A invariants ON PURPOSE:
 *   - L can reach 255 (62 + 200 saturates), so nothing may assume L <= 254;
 *   - S can reach 8*255 = 2040 > 1023, so the Config-A argmin key
 *     (S<<6)|d OVERFLOWS uint16.  argmin here uses a 5-bit key per 32-lane
 *     half-segment and an explicit cross-half pick (lower half wins ties,
 *     which IS "lowest d wins").
 */
#ifndef SGMMS_KERNELS_H
#define SGMMS_KERNELS_H

#ifndef SGM_D
#define SGM_D 64
#endif
#if SGM_D != 64
#error "Configuration B pins D=64"
#endif
#define P2 200
#define COST_INVALID 63
/* P1 is per-path at runtime; the header default is irrelevant but must exist */
#define P1 20

#ifndef SGM_L2PF
#define SGM_L2PF(a,s,w,h) do{}while(0)
#endif
#ifndef SGM_PCYC
#define SGM_PCYC() 0ULL
#endif

#include "../../sgmdsp/src/sgm_hvx_kernels.h"   /* step2, widen_pair, census
                                                   transpose prims, cost_group,
                                                   pk_t ... (validated) */

#define MSB_P2 200
#define MSB_CINVAL 63
#define CW 9
#define CH 7

static unsigned long long sgm_leftover = 0;   /* scalar-fallback pixel count */

static inline int msb_clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
static inline unsigned char msb_sat8(unsigned s){ return s>255u?255u:(unsigned char)s; }

/* per-path P1: patch the packed constants after pk_init() */
static inline pk_t pk_b(int p1)
{
    pk_t k = pk_init();
    k.p1 = SPL(p1);
    return k;
}

/* ==== horizontal Sobel prefilter ========================================= */
/* reference: gx = -a[ym][xm]+a[ym][xp] -2a[y][xm]+2a[y][xp] -a[yp][xm]+a[yp][xp]
 *            out = clamp(128 + (gx>>3), 0, 255),  x/y clamped to the image.
 * Factored as gx = T(xp) - T(xm) with T(x) = a[ym][x] + 2 a[y][x] + a[yp][x].
 * T is exact in uint16 (<= 1020); gx>>3 is an ARITHMETIC shift; the result is
 * always in [0,255] so the byte pack is exact (low byte == value). */
static void sobel_rows(const unsigned char *im, unsigned char *out,
                       int W, int H, int y0, int y1)
{
    for (int y = y0; y < y1; y++)
        for (int x = 0; x < W; x++) {
            int xm=msb_clampi(x-1,0,W-1), xp=msb_clampi(x+1,0,W-1);
            int ym=msb_clampi(y-1,0,H-1), yp=msb_clampi(y+1,0,H-1);
            int gx = -im[ym*W+xm]+im[ym*W+xp]-2*im[y*W+xm]+2*im[y*W+xp]
                     -im[yp*W+xm]+im[yp*W+xp];
            out[y*W+x]=(unsigned char)msb_clampi(128+(gx>>3),0,255);
        }
}

/* T: caller-provided halfword row buffer with >=1 guard halfword on the left
 * and >=64+1 on the right of [0,W); 128-byte-aligned base is T-64. */
static void sobel_rows_hvx(const unsigned char *im, unsigned char *out,
                           int W, int H, int y0, int y1, short *T)
{
    const HVX_Vector h128 = Q6_V_vsplat_R(0x00800080);
    for (int y = y0; y < y1; y++) {
        const unsigned char *rm = im + (size_t)msb_clampi(y-1,0,H-1)*W;
        const unsigned char *rc = im + (size_t)y*W;
        const unsigned char *rp = im + (size_t)msb_clampi(y+1,0,H-1)*W;
        int x = 0;
        for (;;) {                                   /* pass 1: T(x) */
            HVX_VectorPair a = widen_pair(*(const HVX_UVector *)(rm + x));
            HVX_VectorPair b = widen_pair(*(const HVX_UVector *)(rc + x));
            HVX_VectorPair c = widen_pair(*(const HVX_UVector *)(rp + x));
            HVX_Vector t0 = Q6_Vh_vadd_VhVh(Q6_V_lo_W(a), Q6_V_lo_W(c));
            t0 = Q6_Vh_vadd_VhVh(t0, Q6_Vh_vadd_VhVh(Q6_V_lo_W(b), Q6_V_lo_W(b)));
            HVX_Vector t1 = Q6_Vh_vadd_VhVh(Q6_V_hi_W(a), Q6_V_hi_W(c));
            t1 = Q6_Vh_vadd_VhVh(t1, Q6_Vh_vadd_VhVh(Q6_V_hi_W(b), Q6_V_hi_W(b)));
            *(HVX_UVector *)(T + x)      = t0;
            *(HVX_UVector *)(T + x + 64) = t1;
            if (x + 128 >= W) break;
            x += 128;
            if (x + 128 > W) x = W - 128;            /* overlapping tail block */
        }
        T[-1] = T[0]; T[W] = T[W-1];                 /* clamped-x guards */
        x = 0;
        for (;;) {                                   /* pass 2: pack gx */
            HVX_Vector l0 = *(const HVX_UVector *)(T + x - 1);
            HVX_Vector l1 = *(const HVX_UVector *)(T + x - 1 + 64);
            HVX_Vector r0 = *(const HVX_UVector *)(T + x + 1);
            HVX_Vector r1 = *(const HVX_UVector *)(T + x + 1 + 64);
            HVX_Vector d0 = Q6_Vh_vadd_VhVh(Q6_Vh_vasr_VhR(Q6_Vh_vsub_VhVh(r0,l0),3),h128);
            HVX_Vector d1 = Q6_Vh_vadd_VhVh(Q6_Vh_vasr_VhR(Q6_Vh_vsub_VhVh(r1,l1),3),h128);
            *(HVX_UVector *)(out + (size_t)y*W + x) = Q6_Vb_vpacke_VhVh(d1, d0);
            if (x + 128 >= W) break;
            x += 128;
            if (x + 128 > W) x = W - 128;
        }
    }
}

/* ==== 2x2 box-mean downsample ============================================ */
static void half_rows(const unsigned char *im, unsigned char *out,
                      int W, int H, int y0h, int y1h)
{
    int w = W/2; (void)H;
    for (int y = y0h; y < y1h; y++)
        for (int x = 0; x < w; x++)
            out[(size_t)y*w+x]=(unsigned char)((im[(size_t)(2*y)*W+2*x]+im[(size_t)(2*y)*W+2*x+1]
                                 +im[(size_t)(2*y+1)*W+2*x]+im[(size_t)(2*y+1)*W+2*x+1]+2)>>2);
}
static void half_rows_hvx(const unsigned char *im, unsigned char *out,
                          int W, int H, int y0h, int y1h)
{
    int w = W/2; (void)H;
    const HVX_Vector two = Q6_V_vsplat_R(0x00020002);
    for (int y = y0h; y < y1h; y++) {
        const unsigned char *r0 = im + (size_t)(2*y)*W;
        const unsigned char *r1 = im + (size_t)(2*y+1)*W;
        int x = 0;
        for (;;) {
            HVX_Vector s0 = Q6_Vh_vadd_VhVh(
                Q6_Vh_vdmpy_VubRb(*(const HVX_UVector *)(r0 + 2*x), 0x01010101),
                Q6_Vh_vdmpy_VubRb(*(const HVX_UVector *)(r1 + 2*x), 0x01010101));
            HVX_Vector s1 = Q6_Vh_vadd_VhVh(
                Q6_Vh_vdmpy_VubRb(*(const HVX_UVector *)(r0 + 2*x + 128), 0x01010101),
                Q6_Vh_vdmpy_VubRb(*(const HVX_UVector *)(r1 + 2*x + 128), 0x01010101));
            s0 = Q6_Vuh_vlsr_VuhR(Q6_Vh_vadd_VhVh(s0, two), 2);
            s1 = Q6_Vuh_vlsr_VuhR(Q6_Vh_vadd_VhVh(s1, two), 2);
            *(HVX_UVector *)(out + (size_t)y*w + x) = Q6_Vb_vpacke_VhVh(s1, s0);
            if (x + 128 >= w) break;
            x += 128;
            if (x + 128 > w) x = w - 128;            /* overlapping tail block */
        }
    }
}

/* ==== census 9x7 (verbatim from Config A, validated) ===================== */
static void census_rows(const unsigned char *im, unsigned long long *out,
                        int W, int H, int y0, int y1)
{
    for (int y = y0; y < y1; y++)
        for (int x = 0; x < W; x++) {
            unsigned char c = im[(size_t)y*W + x];
            unsigned long long w = 0; int n = 0;
            for (int dy = -(CH/2); dy <= CH/2; dy++)
                for (int dx = -(CW/2); dx <= CW/2; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int yy = msb_clampi(y+dy, 0, H-1), xx = msb_clampi(x+dx, 0, W-1);
                    if (im[(size_t)yy*W + xx] < c) w |= (1ULL << n);
                    n++;
                }
            out[(size_t)y*W + x] = w;
        }
}
static const signed char CNS_DY[62] = {
 -3,-3,-3,-3,-3,-3,-3,-3,-3, -2,-2,-2,-2,-2,-2,-2,-2,-2, -1,-1,-1,-1,-1,-1,-1,-1,-1,
  0, 0, 0, 0,    0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 1,  2, 2, 2, 2, 2, 2, 2, 2, 2,
  3, 3, 3, 3, 3, 3, 3, 3, 3 };
static const signed char CNS_DX[62] = {
 -4,-3,-2,-1, 0, 1, 2, 3, 4, -4,-3,-2,-1, 0, 1, 2, 3, 4, -4,-3,-2,-1, 0, 1, 2, 3, 4,
 -4,-3,-2,-1,    1, 2, 3, 4, -4,-3,-2,-1, 0, 1, 2, 3, 4, -4,-3,-2,-1, 0, 1, 2, 3, 4,
 -4,-3,-2,-1, 0, 1, 2, 3, 4 };

static void census_hvx_block(const unsigned char *im, unsigned long long *out,
                             int w, int h, int y, int x0)
{
    HVX_Vector ctr = *(const HVX_UVector *)(im + (size_t)y*w + x0);
    HVX_Vector zero = Q6_V_vzero();
    HVX_Vector plane[8];
    for (int p = 0; p < 8; p++) {
        HVX_Vector a = zero;
        int cnt = (p == 7) ? 6 : 8;
        for (int b = 0; b < cnt; b++) {
            int k = p*8 + b;
            const unsigned char *row = im + (size_t)msb_clampi(y + CNS_DY[k], 0, h-1)*w;
            HVX_Vector nb = *(const HVX_UVector *)(row + x0 + CNS_DX[k]);
            a = Q6_Vb_condacc_QVbVb(Q6_Q_vcmp_gt_VubVub(ctr, nb), a,
                                    Q6_V_vsplat_R((int)((1u<<b)*0x01010101u)));
        }
        plane[p] = a;
    }
    HVX_VectorPair a0=Q6_W_vshuff_VVR(plane[1],plane[0],-1);
    HVX_VectorPair a1=Q6_W_vshuff_VVR(plane[3],plane[2],-1);
    HVX_VectorPair a2=Q6_W_vshuff_VVR(plane[5],plane[4],-1);
    HVX_VectorPair a3=Q6_W_vshuff_VVR(plane[7],plane[6],-1);
    HVX_VectorPair b0=Q6_W_vshuff_VVR(Q6_V_lo_W(a1),Q6_V_lo_W(a0),-2);
    HVX_VectorPair b1=Q6_W_vshuff_VVR(Q6_V_hi_W(a1),Q6_V_hi_W(a0),-2);
    HVX_VectorPair b2=Q6_W_vshuff_VVR(Q6_V_lo_W(a3),Q6_V_lo_W(a2),-2);
    HVX_VectorPair b3=Q6_W_vshuff_VVR(Q6_V_hi_W(a3),Q6_V_hi_W(a2),-2);
    HVX_VectorPair c0=Q6_W_vshuff_VVR(Q6_V_lo_W(b2),Q6_V_lo_W(b0),-4);
    HVX_VectorPair c1=Q6_W_vshuff_VVR(Q6_V_hi_W(b2),Q6_V_hi_W(b0),-4);
    HVX_VectorPair c2=Q6_W_vshuff_VVR(Q6_V_lo_W(b3),Q6_V_lo_W(b1),-4);
    HVX_VectorPair c3=Q6_W_vshuff_VVR(Q6_V_hi_W(b3),Q6_V_hi_W(b1),-4);
    HVX_UVector *o = (HVX_UVector *)(out + (size_t)y*w + x0);
    o[0]=Q6_V_lo_W(c0); o[1]=Q6_V_hi_W(c0);
    o[2]=Q6_V_lo_W(c1); o[3]=Q6_V_hi_W(c1);
    o[4]=Q6_V_lo_W(c2); o[5]=Q6_V_hi_W(c2);
    o[6]=Q6_V_lo_W(c3); o[7]=Q6_V_hi_W(c3);
}
static void census_span(const unsigned char *im, unsigned long long *out,
                        int w, int h, int y, int x0, int x1)
{
    for (int x = x0; x < x1; x++) {
        unsigned char c = im[(size_t)y*w + x];
        unsigned long long v = 0; int n = 0;
        for (int dy = -(CH/2); dy <= CH/2; dy++)
            for (int dx = -(CW/2); dx <= CW/2; dx++) {
                if (dx == 0 && dy == 0) continue;
                int yy = msb_clampi(y+dy,0,h-1), xx = msb_clampi(x+dx,0,w-1);
                if (im[(size_t)yy*w + xx] < c) v |= (1ULL << n);
                n++;
            }
        out[(size_t)y*w + x] = v;
    }
}
static void census_rows_hvx(const unsigned char *im, unsigned long long *out,
                            int w, int h, int y0, int y1)
{
    const int G = CW/2;
    for (int y = y0; y < y1; y++) {
        int x = G;
        for (; x + 128 <= w - G; x += 128) census_hvx_block(im, out, w, h, y, x);
        if (x < w - G && w - G - 128 >= G) {
            census_hvx_block(im, out, w, h, y, w - G - 128);
            x = w - G;
        }
        census_span(im, out, w, h, y, 0, G);
        census_span(im, out, w, h, y, x, w);
    }
}

/* ==== cost volume, with OPTIONAL fused multiscale merge ==================
 * merged(x,y,d) = (c2[d] + c1_at(x/2,y/2)[d/2] + 1) >> 1, applied AFTER the
 * x-d<0 -> 63 rule (the reference averages the invalid 63s too, and a
 * full-res-invalid lane can meet a VALID half-res cost, e.g. x=6,d=7).
 *
 * Expansion of a 64-byte C1 record r to [r0,r0,r1,r1,...] uses the vshuff
 * self-interleave: lo(vshuff(v,v,-1)) duplicates lanes 0..63 pairwise, whose
 * FIRST 64 bytes are exactly r[0..31] each twice; lo(vshuff(t,t,-64)) then
 * broadcasts those 64 bytes to both segments (pixels x and x+1 share the same
 * half-res pixel when x is even, which it always is at PPV=2).
 * (a+b+1)>>1 is exactly Q6_Vub_vavg_VubVub_rnd. */
static inline HVX_Vector merge_c1(HVX_Vector v, const unsigned char *c1row, int x)
{
    HVX_Vector c1v = *(const HVX_UVector *)(c1row + (size_t)(x >> 1)*DPAD);
    HVX_Vector t = Q6_V_lo_W(Q6_W_vshuff_VVR(c1v, c1v, -1));
    HVX_Vector e = Q6_V_lo_W(Q6_W_vshuff_VVR(t, t, -64));
    return Q6_Vub_vavg_VubVub_rnd(v, e);
}
static inline void merge_c1_scalar(unsigned char *c, const unsigned char *c1p)
{
    for (int d = 0; d < SGM_D; d++)
        c[d] = (unsigned char)((c[d] + c1p[d>>1] + 1) >> 1);
}

static unsigned long long sgm_prolog[16];

/* c1 == NULL: plain cost volume (stage 1).  c1 != NULL: fused merge with the
 * half-res cost volume whose row pitch is hw records (stage 2). */
static void build_cost_rows_hvx_b(const unsigned long long *cl, const unsigned long long *cr,
                                  unsigned char *CV, int W, int H, int y0, int y1, int slot,
                                  const unsigned char *c1, int hw)
{
    const pk_t k = pk_init();
    unsigned long long pro = 0;
    const int XS = DPAD;
    int XV = (XS < W) ? XS : W; XV -= XV % PPV;
    const HVX_Vector dvv = lane_dindex(0), pvv = lane_poffset();
    for (int y = y0; y < y1; y++) {
        const unsigned long long *clr = cl + (size_t)y*W, *crr = cr + (size_t)y*W;
        unsigned char *CVr = CV + (size_t)y*W*DPAD;
        const unsigned char *c1row = c1 ? c1 + (size_t)(y >> 1)*hw*DPAD : 0;
        unsigned long long tp0 = SGM_PCYC();
        for (int x = 0; x < XV; x += PPV) {
            HVX_Vector v = cost_group_edge(clr, crr, x, k.pad, dvv, pvv);
            if (c1row) v = merge_c1(v, c1row, x);
            *(HVX_Vector *)(CVr + (size_t)x*DPAD) = v;
        }
        for (int x = XV; x < XS && x < W; x++) {         /* PPV leftovers, none at even W */
            unsigned char *c = CVr + (size_t)x*DPAD;
            unsigned long long a = clr[x];
            int dmax = (x < SGM_D-1) ? x : SGM_D-1;
            for (int d = 0; d <= dmax; d++) c[d] = (unsigned char)__builtin_popcountll(a ^ crr[x-d]);
            for (int d = dmax+1; d < SGM_D; d++) c[d] = MSB_CINVAL;
            if (c1row) merge_c1_scalar(c, c1row + (size_t)(x>>1)*DPAD);
        }
        pro += SGM_PCYC() - tp0;
        int x = (XS < W) ? XS : W;
        for (; x + PPV <= W; x += PPV) {
            HVX_Vector v = cost_group(clr, crr, x, k.pad);
            if (c1row) v = merge_c1(v, c1row, x);
            *(HVX_Vector *)(CVr + (size_t)x*DPAD) = v;
        }
        for (; x < W; x++) {
            unsigned char *c = CVr + (size_t)x*DPAD;
            unsigned long long a = clr[x];
            for (int d = 0; d < SGM_D; d++) c[d] = (unsigned char)__builtin_popcountll(a ^ crr[x-d]);
            if (c1row) merge_c1_scalar(c, c1row + (size_t)(x>>1)*DPAD);
        }
    }
    if (slot >= 0 && slot < 16) sgm_prolog[slot] = pro;
}

/* scalar cost volume (use_hvx==0 cross-check) */
static void build_cost_rows_b(const unsigned long long *cl, const unsigned long long *cr,
                              unsigned char *CV, int W, int H, int y0, int y1,
                              const unsigned char *c1, int hw)
{
    (void)H;
    for (int y = y0; y < y1; y++)
        for (int x = 0; x < W; x++) {
            unsigned char *c = CV + ((size_t)y*W + x)*DPAD;
            unsigned long long a = cl[(size_t)y*W + x];
            for (int d = 0; d < SGM_D; d++) {
                int xr = x - d;
                c[d] = (xr < 0) ? MSB_CINVAL
                     : (unsigned char)__builtin_popcountll(a ^ cr[(size_t)y*W + xr]);
            }
            if (c1) merge_c1_scalar(c, c1 + ((size_t)(y>>1)*hw + (x>>1))*DPAD);
        }
}

/* ==== scalar aggregation step (per-path p1, saturating) ================== */
static inline void step_p1(const unsigned char *Lp, const unsigned char *C,
                           unsigned char *Lo, int p1)
{
    unsigned char mn = 255;
    for (int d = 0; d < SGM_D; d++) if (Lp[d] < mn) mn = Lp[d];
    unsigned p2t = msb_sat8((unsigned)mn + MSB_P2);
    for (int d = 0; d < SGM_D; d++) {
        unsigned m = Lp[d], t;
        t = (d == 0)       ? 255u : msb_sat8((unsigned)Lp[d-1] + p1); if (t < m) m = t;
        t = (d == SGM_D-1) ? 255u : msb_sat8((unsigned)Lp[d+1] + p1); if (t < m) m = t;
        if (p2t < m) m = p2t;
        Lo[d] = msb_sat8((unsigned)C[d] + (unsigned char)(m - mn));
    }
}

/* scalar 8-path aggregation, direct port of the oracle (S must be zeroed) */
static const int MSB_DIRS8[8][3]={
    { 1,0,40},{-1,0,40},{0, 1,20},{0,-1,20},
    { 1,1,10},{-1,-1,10},{1,-1,10},{-1,1,10}};
static void aggregate_scalar_b(const unsigned char *C, unsigned short *S,
                               int W, int H, unsigned char *La, unsigned char *Lb)
{
    for (int p = 0; p < 8; p++) {
        int dx=MSB_DIRS8[p][0], dy=MSB_DIRS8[p][1], p1=MSB_DIRS8[p][2];
        if (dy == 0) {
            for (int y = 0; y < H; y++) {
                unsigned char Lrow[2][SGM_D]; int cb = 0;
                int x0=dx<0?W-1:0, x1=dx<0?-1:W, xs=dx<0?-1:1;
                for (int x = x0; x != x1; x += xs) {
                    const unsigned char *c = C + ((size_t)y*W + x)*DPAD;
                    unsigned char *cur = Lrow[cb];
                    if (x != x0) step_p1(Lrow[cb^1], c, cur, p1);
                    else memcpy(cur, c, SGM_D);
                    unsigned short *s = S + ((size_t)y*W + x)*DPAD;
                    for (int d = 0; d < SGM_D; d++) s[d] += cur[d];
                    cb ^= 1;
                }
            }
        } else {
            int y0=dy<0?H-1:0, y1=dy<0?-1:H, ys=dy<0?-1:1;
            memset(La, 0, (size_t)W*SGM_D);
            for (int y = y0; y != y1; y += ys) {
                for (int x = 0; x < W; x++) {
                    const unsigned char *c = C + ((size_t)y*W + x)*DPAD;
                    unsigned char *cur = Lb + (size_t)x*SGM_D;
                    int px = x-dx, py = y-dy;
                    if (px >= 0 && px < W && py >= 0 && py < H)
                        step_p1(La + (size_t)px*SGM_D, c, cur, p1);
                    else memcpy(cur, c, SGM_D);
                    unsigned short *s = S + ((size_t)y*W + x)*DPAD;
                    for (int d = 0; d < SGM_D; d++) s[d] += cur[d];
                }
                unsigned char *t = La; La = Lb; Lb = t;
            }
        }
    }
}

/* ==== HVX vertical paths (Config A shape + per-path p1) ================== */
static void path_leftover_b(const unsigned char *CV, unsigned short *S, int W, int H,
                            int dirx, int diry, unsigned char *Lrow, int r0, int r1,
                            int first, int p1)
{
    unsigned char Ltmp[DPAD], Lst[DPAD];
    if (r1 <= r0) return;
    if (diry == 0) {
        for (int y = r0; y < r1; y++) {
            int x0 = (dirx > 0) ? 0 : W-1;
            for (int i = 0; i < W; i++) {
                int x = x0 + dirx*i;
                const unsigned char *C = CV + ((size_t)y*W + x)*DPAD;
                if (i == 0) memcpy(Ltmp, C, DPAD); else step_p1(Lst, C, Ltmp, p1);
                memcpy(Lst, Ltmp, DPAD);
                unsigned short *Sp = S + ((size_t)y*W + x)*DPAD;
                for (int d = 0; d < DPAD; d++)
                    Sp[d] = first ? Ltmp[d] : (unsigned short)(Sp[d] + Ltmp[d]);
            }
            sgm_leftover += W;
        }
    } else {
        int y0 = (diry > 0) ? 0 : H-1;
        for (int j = 0; j < H; j++) {
            int y = y0 + diry*j;
            for (int x = r0; x < r1; x++) {
                const unsigned char *C = CV + ((size_t)y*W + x)*DPAD;
                unsigned char *Lp = Lrow + (size_t)x*DPAD;
                if (j == 0) memcpy(Ltmp, C, DPAD); else step_p1(Lp, C, Ltmp, p1);
                memcpy(Lp, Ltmp, DPAD);
                unsigned short *Sp = S + ((size_t)y*W + x)*DPAD;
                for (int d = 0; d < DPAD; d++)
                    Sp[d] = first ? Ltmp[d] : (unsigned short)(Sp[d] + Ltmp[d]);
            }
        }
        sgm_leftover += (unsigned long long)(r1-r0)*H;
    }
}

static void path_vert2_b(const unsigned char *CV, unsigned short *S, int W, int H,
                         int diry, unsigned char *Lrow, int r0, int r1, int first, int p1)
{
    const pk_t k = pk_b(p1);
    int y0 = (diry > 0) ? 0 : H-1;
    int xe = r0 + ((r1 - r0) / PPV) * PPV;
    const int PFC = 64;
    for (int j = 0; j < H; j++) {
        int y = y0 + diry*j;
        const unsigned char *CVr = CV + (size_t)y*W*DPAD;
        unsigned short     *Sr   = S  + (size_t)y*W*DPAD;
        for (int xc = r0; xc < xe; xc += PFC) {
            int xn = xc + PFC; if (xn > xe) xn = xe;
            if (xn < xe)
                SGM_L2PF(Sr + (size_t)xn*DPAD, 128, 128, (2*DPAD*PFC)/128);
            int x = xc;
            for (; x + 4*PPV <= xn; x += 4*PPV) {
                const unsigned char *cp = CVr + (size_t)x*DPAD;
                unsigned char *lp = Lrow + (size_t)x*DPAD;
                unsigned short *sp = Sr + (size_t)x*DPAD;
                HVX_Vector C0 = *(const HVX_Vector *)(cp);
                HVX_Vector C1 = *(const HVX_Vector *)(cp + 128);
                HVX_Vector C2 = *(const HVX_Vector *)(cp + 256);
                HVX_Vector C3 = *(const HVX_Vector *)(cp + 384);
                HVX_Vector L0, L1, L2, L3;
                if (j == 0) { L0=C0; L1=C1; L2=C2; L3=C3; }
                else {
                    L0 = step2(*(HVX_Vector *)(lp),       C0, &k);
                    L1 = step2(*(HVX_Vector *)(lp + 128), C1, &k);
                    L2 = step2(*(HVX_Vector *)(lp + 256), C2, &k);
                    L3 = step2(*(HVX_Vector *)(lp + 384), C3, &k);
                }
                *(HVX_Vector *)(lp)       = L0;
                *(HVX_Vector *)(lp + 128) = L1;
                *(HVX_Vector *)(lp + 256) = L2;
                *(HVX_Vector *)(lp + 384) = L3;
                acc2(L0, (HVX_Vector *)(sp),       first);
                acc2(L1, (HVX_Vector *)(sp + 128), first);
                acc2(L2, (HVX_Vector *)(sp + 256), first);
                acc2(L3, (HVX_Vector *)(sp + 384), first);
            }
            for (; x < xn; x += PPV) {
                HVX_Vector C = *(const HVX_Vector *)(CVr + (size_t)x*DPAD);
                HVX_Vector *Lp = (HVX_Vector *)(Lrow + (size_t)x*DPAD);
                HVX_Vector L = (j == 0) ? C : step2(*Lp, C, &k);
                *Lp = L;
                acc2(L, (HVX_Vector *)(Sr + (size_t)x*DPAD), first);
            }
        }
    }
    if (xe < r1) path_leftover_b(CV, S, W, H, 0, diry, Lrow, xe, r1, first, p1);
}

/* ==== HVX horizontal paths (Config A shape + per-path p1) ================ */
static void path_horiz2_b(const unsigned char *CV, unsigned short *S, int W, int H,
                          int dirx, int r0, int r1, int first, int p1)
{
    const int pf  = first ? 2 : 1;
    const pk_t k  = pk_b(p1);
    const int PFC = 64;
    int x0 = (dirx > 0) ? 0 : W-1;
    int y = r0;
    int ye = r0 + ((r1 - r0) / PPV) * PPV;
    if (W % HB) { path_leftover_b(CV,S,W,H,dirx,0,0,r0,r1,first,p1); return; }
    for (; y < ye; y += PPV) {
        const unsigned char *A[PPV]; unsigned short *Sr[PPV];
        for (int i = 0; i < PPV; i++) {
            A[i]  = CV + (size_t)(y+i)*W*DPAD;
            Sr[i] = S  + (size_t)(y+i)*W*DPAD;
        }
        HVX_Vector L = k.big;
        HVX_VectorPair wb[HB]; (void)wb;
        for (int ic = 0; ic < W; ic += PFC) {
            int inx = ic + PFC; if (inx > W) inx = W;
            if (inx < W) {
                int xp = x0 + dirx*(dirx > 0 ? inx : inx + PFC - 1);
                if (xp >= 0 && xp + PFC <= W) {
                    for (int i = 0; i < PPV; i++) {
                        if (pf == 1) SGM_L2PF(Sr[i] + (size_t)xp*DPAD, 128, 128, (2*DPAD*PFC)/128);
                        else         SGM_L2PF(A[i]  + (size_t)xp*DPAD, 128, 128, (DPAD*PFC)/128);
                    }
                }
            }
            for (int i = ic; i < inx; i++) {
                int x = x0 + dirx*i;
                HVX_Vector C = hz_cost(A, (size_t)x*DPAD);
                L = (i == 0) ? C : step2(L, C, &k);
                HVX_VectorPair w = widen_pair(L);
                hz_store(Sr, (size_t)x*DPAD*2, &w, first);
            }
        }
    }
    if (ye < r1) path_leftover_b(CV, S, W, H, dirx, 0, 0, ye, r1, first, p1);
}

static void path_horiz_fused_b(const unsigned char *CV, unsigned short *S, int W, int H,
                               int r0, int r1, unsigned char *rowbuf, int p1)
{
    const pk_t k = pk_b(p1);
    const int PFC = 64;
    int ye = r0 + ((r1 - r0) / PPV) * PPV;
    if (W % HB) {
        path_horiz2_b(CV, S, W, H,  1, r0, r1, 1, p1);
        path_horiz2_b(CV, S, W, H, -1, r0, r1, 0, p1);
        return;
    }
    for (int y = r0; y < ye; y += PPV) {
        const unsigned char *A[PPV]; unsigned short *Sr[PPV];
        for (int i = 0; i < PPV; i++) {
            A[i]  = CV + (size_t)(y+i)*W*DPAD;
            Sr[i] = S  + (size_t)(y+i)*W*DPAD;
        }
        HVX_Vector L = k.big;
        for (int ic = 0; ic < W; ic += PFC) {
            int inx = ic + PFC; if (inx > W) inx = W;
            if (inx + PFC <= W)
                for (int i = 0; i < PPV; i++)
                    SGM_L2PF(A[i] + (size_t)inx*DPAD, 128, 128, (DPAD*PFC)/128);
            for (int x = ic; x < inx; x++) {
                HVX_Vector C = hz_cost(A, (size_t)x*DPAD);
                L = (x == 0) ? C : step2(L, C, &k);
                *(HVX_Vector *)(rowbuf + (size_t)x*128) = L;
            }
        }
        L = k.big;
        for (int ic = W - 1; ic >= 0; ic -= PFC) {
            int inx = ic - PFC; if (inx < -1) inx = -1;
            if (inx - PFC >= -1)
                for (int i = 0; i < PPV; i++)
                    SGM_L2PF(A[i] + (size_t)(inx-PFC+1)*DPAD, 128, 128, (DPAD*PFC)/128);
            for (int x = ic; x > inx; x--) {
                HVX_Vector C = hz_cost(A, (size_t)x*DPAD);
                L = (x == W-1) ? C : step2(L, C, &k);
                HVX_VectorPair w = widen_add_pair(L, *(const HVX_Vector *)(rowbuf + (size_t)x*128));
                hz_store(Sr, (size_t)x*DPAD*2, &w, 1);
            }
        }
    }
    if (ye < r1) {
        path_horiz2_b(CV, S, W, H,  1, ye, r1, 1, p1);
        path_horiz2_b(CV, S, W, H, -1, ye, r1, 0, p1);
    }
}

/* ==== HVX diagonal paths ==================================================
 * A diagonal path's dependence chains are its DIAGONAL LINES: pixel (x,y)
 * depends on (x-dx, y-dy), i.e. the SAME chain one processed-row earlier.
 * Chain id v = x - dirx*j (j = processed-row index; j=0 is the first row the
 * sweep visits).  A worker owns a band of chains [c0,c1); at row j its pixels
 * are columns [c0+dirx*j, c1+dirx*j) clipped to [0,W).  Bands partition every
 * row's pixels, so S accumulation is worker-disjoint with NO synchronisation.
 * Bands are cut at equal PIXEL counts (edge chains are short).
 *
 * L state is indexed by CHAIN SLOT s = v - c0, not by column: a chain's
 * previous element is the SAME slot one row earlier, so the update is a pure
 * in-place L[s] = step2(L[s], C(x)) with x = s + c0 + dirx*j.  This keeps
 * every L access 128-aligned (vectors start at even s; only the cost-volume
 * read is ever unaligned) and needs no double buffer.  The buffer starts at
 * 255: a 255 record makes step2 return C exactly (mn=255, every candidate
 * 255), which IS the reference's "no previous pixel -> L = C" rule for the
 * first row and for chains entering at an image edge; a chain's slot is
 * untouched until the row it enters.
 *
 * Traps already paid for, honoured here: the row-state indexing differs from
 * the vertical path, and P1 is per-path (10 on diagonals).  The 4-vector
 * unroll is the same latency-hiding that bought 2.5x on the vertical paths.
 */
static void path_diag_b(const unsigned char *CV, unsigned short *S, int W, int H,
                        int dirx, int diry, int p1,
                        unsigned char *Lb, int c0, int c1)
{
    const pk_t k = pk_b(p1);
    int nc = c1 - c0;
    memset(Lb, 255, (size_t)(nc + 2)*64);
    const int PFC = 64;
    for (int j = 0; j < H; j++) {
        int y = (diry > 0) ? j : (H-1-j);
        int xoff = c0 + dirx*j;
        int slo = (0 > -xoff) ? 0 : -xoff;
        int shi = (nc < W - xoff) ? nc : W - xoff;
        if (slo >= shi) continue;
        const unsigned char *CVr = CV + ((size_t)y*W + xoff)*DPAD;   /* by s */
        unsigned short     *Sr   = S  + ((size_t)y*W + xoff)*DPAD;   /* by s */
        int s = slo;
        if (s & 1) {                       /* entering chain at an odd slot */
            HVX_Vector C  = *(const HVX_UVector *)(CVr + (size_t)s*DPAD);
            HVX_Vector L  = step2(*(const HVX_UVector *)(Lb + (size_t)s*64), C, &k);
            unsigned char tmp[128] __attribute__((aligned(128)));
            *(HVX_Vector *)tmp = L;
            memcpy(Lb + (size_t)s*64, tmp, 64);      /* 64B: slot s only */
            HVX_VectorPair q = widen_pair(L);
            HVX_Vector *Sp = (HVX_Vector *)(Sr + (size_t)s*DPAD);
            Sp[0] = Q6_Vh_vadd_VhVh(Sp[0], Q6_V_lo_W(q));
            s++;
        }
        for (int xc = s; xc < shi; xc += PFC) {
            int xn = xc + PFC; if (xn > shi) xn = shi;
            if (xn < shi) {
                int left = shi - xn; if (left > PFC) left = PFC;
                SGM_L2PF(Sr + (size_t)xn*DPAD, 128, 128, (2*DPAD*(unsigned)left)/128);
            }
            for (; s + 8 <= xn; s += 8) {            /* 4 chains in flight */
                const unsigned char *cp = CVr + (size_t)s*DPAD;
                unsigned char *lp = Lb + (size_t)s*64;
                unsigned short *sp = Sr + (size_t)s*DPAD;
                HVX_Vector C0 = *(const HVX_UVector *)(cp);
                HVX_Vector C1 = *(const HVX_UVector *)(cp + 128);
                HVX_Vector C2 = *(const HVX_UVector *)(cp + 256);
                HVX_Vector C3 = *(const HVX_UVector *)(cp + 384);
                HVX_Vector L0 = step2(*(HVX_Vector *)(lp),       C0, &k);
                HVX_Vector L1 = step2(*(HVX_Vector *)(lp + 128), C1, &k);
                HVX_Vector L2 = step2(*(HVX_Vector *)(lp + 256), C2, &k);
                HVX_Vector L3 = step2(*(HVX_Vector *)(lp + 384), C3, &k);
                *(HVX_Vector *)(lp)       = L0;
                *(HVX_Vector *)(lp + 128) = L1;
                *(HVX_Vector *)(lp + 256) = L2;
                *(HVX_Vector *)(lp + 384) = L3;
                acc2(L0, (HVX_Vector *)(sp),       0);
                acc2(L1, (HVX_Vector *)(sp + 128), 0);
                acc2(L2, (HVX_Vector *)(sp + 256), 0);
                acc2(L3, (HVX_Vector *)(sp + 384), 0);
            }
            for (; s + 2 <= xn; s += 2) {
                HVX_Vector C = *(const HVX_UVector *)(CVr + (size_t)s*DPAD);
                HVX_Vector *Lp = (HVX_Vector *)(Lb + (size_t)s*64);
                HVX_Vector L = step2(*Lp, C, &k);
                *Lp = L;
                acc2(L, (HVX_Vector *)(Sr + (size_t)s*DPAD), 0);
            }
            if (s < xn) {                  /* trailing single (odd count) */
                HVX_Vector C  = *(const HVX_UVector *)(CVr + (size_t)s*DPAD);
                HVX_Vector L  = step2(*(const HVX_UVector *)(Lb + (size_t)s*64), C, &k);
                unsigned char tmp[128] __attribute__((aligned(128)));
                *(HVX_Vector *)tmp = L;
                memcpy(Lb + (size_t)s*64, tmp, 64);
                HVX_VectorPair q = widen_pair(L);
                HVX_Vector *Sp = (HVX_Vector *)(Sr + (size_t)s*DPAD);
                Sp[0] = Q6_Vh_vadd_VhVh(Sp[0], Q6_V_lo_W(q));
                s++;
            }
        }
    }
}

/* Equal-PIXEL chain split for nw workers: chain v (in [v0g, v0g+nc)) has
 * len(v) = #rows where its column is inside the image.  Returns cut points
 * cut[0..nw] with cut[0]=v0g, cut[nw]=v0g+nc. */
static void diag_split(int W, int H, int dirx, int nw, int *cut)
{
    int v0g = (dirx > 0) ? -(H-1) : 0;
    int nc = W + H - 1;
    long long total = (long long)W * H, acc = 0;
    int w = 1;
    cut[0] = v0g; cut[nw] = v0g + nc;
    for (int i = 0; i < nc && w < nw; i++) {
        int v = v0g + i;
        int len;
        if (dirx > 0) { int lo = v < 0 ? -v : 0, hi = (W - v < H) ? (W - v) : H; len = hi - lo; }
        else          { int lo = v - (W-1) > 0 ? v - (W-1) : 0, hi = (v < H) ? v+1 : H; len = hi - lo; }
        if (len < 0) len = 0;
        acc += len;
        while (w < nw && acc >= (total * w) / nw) { cut[w] = v0g + i + 1; w++; }
    }
    while (w < nw) { cut[w] = cut[nw]; w++; }
}

/* ==== argmin, Config-B key ================================================
 * S <= 8*255 = 2040, so the uint16 key is (S<<5)|(d&31) per 32-lane half
 * (2040*32+31 = 65311 fits); the two halves are then compared explicitly,
 * ties to the LOW half = lowest d.  Lane 0 of the result holds the winning d.
 */
static inline HVX_Vector aminb_one(const unsigned short *q, HVX_Vector idx5,
                                   HVX_Vector mk31, HVX_Vector c32)
{
    HVX_Vector k = Q6_V_vor_VV(Q6_Vh_vasl_VhR(*(const HVX_Vector *)q, 5), idx5);
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vdelta_VV(k, SPL(2)));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vdelta_VV(k, SPL(4)));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vdelta_VV(k, SPL(8)));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vdelta_VV(k, SPL(16)));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vdelta_VV(k, SPL(32)));
    HVX_Vector kh = Q6_V_vror_VR(k, 64);
    HVX_VectorPred g = Q6_Q_vcmp_gt_VuhVuh(Q6_Vuh_vlsr_VuhR(k, 5),
                                           Q6_Vuh_vlsr_VuhR(kh, 5));
    return Q6_V_vmux_QVV(g, Q6_Vh_vadd_VhVh(Q6_V_vand_VV(kh, mk31), c32),
                            Q6_V_vand_VV(k, mk31));
}

static inline void aminb_consts(HVX_Vector *idx5, HVX_Vector *mk31, HVX_Vector *c32)
{
    unsigned short it[64] __attribute__((aligned(128)));
    for (int i = 0; i < 64; i++) it[i] = (unsigned short)(i & 31);
    *idx5 = *(const HVX_Vector *)it;
    *mk31 = Q6_V_vsplat_R(0x001f001f);
    *c32  = Q6_V_vsplat_R(0x00200020);
}

/* contiguous pixels [p0,p1) (stage-1 argmin over the whole half-res S) */
static void argmin_b_range(const unsigned short *S, unsigned char *disp,
                           unsigned long long p0, unsigned long long p1)
{
    HVX_Vector idx5, mk31, c32; aminb_consts(&idx5, &mk31, &c32);
    unsigned long long p = p0;
    for (; p + 256 <= p1; p += 256) {
        const unsigned short *A = S + p*DPAD;
        if (p + 512 <= p1) SGM_L2PF(S + (p+256)*DPAD, 128, 128, 4*DPAD);
        HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
        HVX_Vector a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
        for (int i = 0; i < 64; i++) {
            const unsigned short *q = A + (size_t)i*DPAD;
            a0 = Q6_V_valign_VVR(aminb_one(q,                    idx5, mk31, c32), a0, 2);
            a1 = Q6_V_valign_VVR(aminb_one(q + (size_t)64*DPAD,  idx5, mk31, c32), a1, 2);
            a2 = Q6_V_valign_VVR(aminb_one(q + (size_t)128*DPAD, idx5, mk31, c32), a2, 2);
            a3 = Q6_V_valign_VVR(aminb_one(q + (size_t)192*DPAD, idx5, mk31, c32), a3, 2);
        }
        *(HVX_UVector *)(disp + p)       = Q6_Vb_vpacke_VhVh(a1, a0);
        *(HVX_UVector *)(disp + p + 128) = Q6_Vb_vpacke_VhVh(a3, a2);
    }
    for (; p < p1; p++) {
        const unsigned short *Sp = S + p*DPAD;
        unsigned short best = Sp[0]; int bd = 0;
        for (int d = 1; d < SGM_D; d++) if (Sp[d] < best) { best = Sp[d]; bd = d; }
        disp[p] = (unsigned char)bd;
    }
}

/* decimated argmin: out(x,y) = argmin of S at (2x, 2y); output rows [y0,y1).
 * ow = W/2 must be a multiple of 128 or handled by the scalar tail. */
static void argmin_b_dec(const unsigned short *S, unsigned char *disp,
                         int W, int ow, int y0, int y1)
{
    HVX_Vector idx5, mk31, c32; aminb_consts(&idx5, &mk31, &c32);
    for (int y = y0; y < y1; y++) {
        const unsigned short *R = S + (size_t)(2*y)*W*DPAD;
        unsigned char *o = disp + (size_t)y*ow;
        int x = 0;
        for (; x + 128 <= ow; x += 128) {
            const unsigned short *A = R + (size_t)(2*x)*DPAD;
            if (x + 256 <= ow)
                SGM_L2PF(R + (size_t)(2*(x+128))*DPAD, 256, 128, 128);
            HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
            for (int i = 0; i < 64; i++) {
                a0 = Q6_V_valign_VVR(aminb_one(A + (size_t)(2*i)*DPAD,       idx5, mk31, c32), a0, 2);
                a1 = Q6_V_valign_VVR(aminb_one(A + (size_t)(2*(i+64))*DPAD,  idx5, mk31, c32), a1, 2);
            }
            *(HVX_UVector *)(o + x) = Q6_Vb_vpacke_VhVh(a1, a0);
        }
        for (; x < ow; x++) {
            const unsigned short *Sp = R + (size_t)(2*x)*DPAD;
            unsigned short best = Sp[0]; int bd = 0;
            for (int d = 1; d < SGM_D; d++) if (Sp[d] < best) { best = Sp[d]; bd = d; }
            o[x] = (unsigned char)bd;
        }
    }
}

#endif /* SGMMS_KERNELS_H */
