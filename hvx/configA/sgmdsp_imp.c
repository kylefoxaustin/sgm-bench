/* sgmdsp_imp.c — SGM stereo on the Hexagon NSP.
 *
 * Semantics are 95emulator's sgm_params.h, reproduced EXACTLY so the output
 * can be diffed against a golden that already agrees across six execution
 * targets (x86_64, A55, A720, A78C, two Mali GPUs via OpenCL).
 *
 *   census : 9x7, 62 bits, bit n set iff neighbour < centre; n increments over
 *            dy (outer, top->bottom) then dx (inner, left->right), centre
 *            skipped; out-of-image neighbours edge-replicated (clamped).
 *   cost   : C(x,y,d) = popcount(cL(x,y) ^ cR(x-d,y)) if x-d >= 0 else 62.
 *   path   : L(p,d) = C + min(Lp(d), Lp(d-1)+P1, Lp(d+1)+P1, min_k Lp(k)+P2)
 *                       - min_k Lp(k);  saturating adds; d+-1 out of range = 255.
 *            First pixel on a path: L = C.
 *   sum    : S (uint16) over 4 paths (L,R,U,D).
 *   output : argmin_d S, LOWEST d wins ties.
 *
 * THE DISPARITY RANGE IS A BUILD PARAMETER.  -DSGM_D=<16k> selects it; the
 * lane geometry (DPAD, PPV) and every kernel follow from it -- see
 * sgm_hvx_kernels.h, which sim/gen.c validates against a scalar reference for
 * each supported D before anything is deployed.
 */
#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "sgmdsp.h"
#include "AEEStdDef.h"
#include <string.h>
#include <stdlib.h>
#include "worker_pool.h"
#include "HAP_compute_res.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define P1 20
#define P2 192
#define CW 9
#define CH 7
#define CBITS (CW*CH-1)      /* 62 */
#define COST_INVALID CBITS

/* l2fetch descriptor: {dir[63:48], stride[47:32], width[31:16], height[15:0]}
 * (matches CreateL2pfParam / HEXAGON_V64_CREATE_H in the SDK's hexagon_cache.h).
 * The L2 fetch engine takes ONE command at a time -- a second command
 * OVERWRITES the first.  The SDK's own l2fetch() helper in
 * libs/qfe/inc/hvx_internal.h is a STUB that does nothing; do not use it. */
static inline void l2pf(const void *a, unsigned stride, unsigned width, unsigned height)
{
    unsigned long long param = (((unsigned long long)stride)<<32) |
                               (((unsigned long long)width)<<16)  | (unsigned long long)height;
    __asm__ __volatile__ ("l2fetch(%0,%1)" : : "r"(a), "r"(param));
}
#define L2PF(addr, stride, width, height) l2pf((const void*)(addr),(stride),(width),(height))
#define SGM_L2PF(a,s,w,h) l2pf((const void*)(a),(s),(w),(h))

#include "sgm_hvx_kernels.h"

static int sgm_pf   = 0;   /* 0 per-path auto, 1 S, 2 CV, 3 alternate, 4 none */
static int sgm_vch  = 4;   /* vertical chains: 1 or 4 column-groups in flight */
static int sgm_hpf  = 64;  /* horizontal prefetch chunk, pixels                */
static int sgm_fuse = 1;   /* fuse the two horizontal paths into one sweep     */
static int sgm_vtcm = 1;
static int sgm_edge = 1;   /* vectorise build_cost's first DPAD columns */   /* park the fused row buffers in VTCM when possible */
static unsigned char *sgm_rowbuf_base = 0;
static unsigned long  sgm_rowbuf_stride = 0;
/* Pixels that fell through to the scalar leftover handlers.  Reported as
 * phases[13]: a MEASURED run must show 0, otherwise a "vector" number is
 * partly scalar and the phase profile is a lie. */
static unsigned long long sgm_leftover = 0;
/* build_cost's first XS=DPAD columns of every row are SCALAR (d can exceed
 * x there, which needs the 62 sentinel).  That is a fixed per-ROW cost that
 * grows with D, so it is timed per job and reported as phases[14] -- max
 * over jobs, i.e. its contribution to the phase's wall time -- rather than
 * left to be guessed at from a two-point width model. */
static unsigned long long sgm_prolog[16];

static inline int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
static inline unsigned char sat8(unsigned s){ return s>255u?255u:(unsigned char)s; }

/* ---- census (scalar reference) ---- */
static void census_rows(const unsigned char *im, unsigned long long *out, int W, int H, int y0, int y1)
{
    for (int y = y0; y < y1; y++)
        for (int x = 0; x < W; x++) {
            unsigned char c = im[(size_t)y*W + x];
            unsigned long long w = 0; int n = 0;
            for (int dy = -(CH/2); dy <= CH/2; dy++)
                for (int dx = -(CW/2); dx <= CW/2; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int yy = clampi(y+dy, 0, H-1), xx = clampi(x+dx, 0, W-1);
                    if (im[(size_t)yy*W + xx] < c) w |= (1ULL << n);
                    n++;
                }
            out[(size_t)y*W + x] = w;
        }
}

/* ---- scalar aggregation over DPAD lanes.  Identical semantics to step2():
 * the cost volume already carries 255 in lanes [D,DPAD), so the pad lanes
 * stay 255 and the d=DPAD-1 boundary is the vector code's mB. Used only for
 * leftover rows/columns, which is zero pixels at every image size measured. */
static inline void step_pad(const unsigned char *Lp, const unsigned char *C, unsigned char *Lo)
{
    unsigned char mn = 255;
    for (int d = 0; d < DPAD; d++) if (Lp[d] < mn) mn = Lp[d];
    unsigned p2t = sat8((unsigned)mn + P2);
    for (int d = 0; d < DPAD; d++) {
        unsigned m = Lp[d], t;
        t = (d == 0)      ? 255u : sat8((unsigned)Lp[d-1] + P1); if (t < m) m = t;
        t = (d == DPAD-1) ? 255u : sat8((unsigned)Lp[d+1] + P1); if (t < m) m = t;
        if (p2t < m) m = p2t;
        Lo[d] = sat8((unsigned)C[d] + (unsigned char)(m - mn));
    }
}
/* leftover rows (diry==0) or columns (diry!=0), one pixel at a time */
static void path_leftover(const unsigned char *CV, unsigned short *S, int W, int H,
                          int dirx, int diry, unsigned char *Lrow, int r0, int r1, int first)
{
    unsigned char Ltmp[DPAD], Lst[DPAD];
    if (r1 <= r0) return;
    if (diry == 0) {
        for (int y = r0; y < r1; y++) {
            int x0 = (dirx > 0) ? 0 : W-1;
            for (int i = 0; i < W; i++) {
                int x = x0 + dirx*i;
                const unsigned char *C = CV + ((size_t)y*W + x)*DPAD;
                if (i == 0) memcpy(Ltmp, C, DPAD); else { step_pad(Lst, C, Ltmp); }
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
                if (j == 0) memcpy(Ltmp, C, DPAD); else step_pad(Lp, C, Ltmp);
                memcpy(Lp, Ltmp, DPAD);
                unsigned short *Sp = S + ((size_t)y*W + x)*DPAD;
                for (int d = 0; d < DPAD; d++)
                    Sp[d] = first ? Ltmp[d] : (unsigned short)(Sp[d] + Ltmp[d]);
            }
        }
        sgm_leftover += (unsigned long long)(r1-r0)*H;
    }
}

/* ---- fully scalar reference path (use_hvx == 0), stride DPAD, range D ---- */
static inline unsigned char cost_at(const unsigned long long *cl, const unsigned long long *cr,
                                    int W, int y, int x, int d)
{
    int xr = x - d;
    if (xr < 0) return COST_INVALID;
    unsigned long long v = cl[(size_t)y*W + x] ^ cr[(size_t)y*W + xr];
    return (unsigned char)__builtin_popcountll(v);
}
static inline void step_ref(const unsigned char *Lp, const unsigned char *C, unsigned char *Lo)
{
    unsigned char mn = 255;
    for (int d = 0; d < SGM_D; d++) if (Lp[d] < mn) mn = Lp[d];
    unsigned p2t = sat8((unsigned)mn + P2);
    for (int d = 0; d < SGM_D; d++) {
        unsigned m = Lp[d], t;
        t = (d == 0)        ? 255u : sat8((unsigned)Lp[d-1] + P1); if (t < m) m = t;
        t = (d == SGM_D-1)  ? 255u : sat8((unsigned)Lp[d+1] + P1); if (t < m) m = t;
        if (p2t < m) m = p2t;
        Lo[d] = sat8((unsigned)C[d] + (unsigned char)(m - mn));
    }
}
static void path(const unsigned long long *cl, const unsigned long long *cr,
                 unsigned short *S, int W, int H, int dirx, int diry,
                 unsigned char *Lbuf)
{
    unsigned char Ccol[SGM_D], Ltmp[SGM_D];
    if (diry == 0) {
        for (int y = 0; y < H; y++) {
            int x0 = (dirx > 0) ? 0 : W-1;
            for (int i = 0; i < W; i++) {
                int x = x0 + dirx*i;
                for (int d = 0; d < SGM_D; d++) Ccol[d] = cost_at(cl, cr, W, y, x, d);
                if (i == 0) memcpy(Ltmp, Ccol, SGM_D);
                else step_ref(Lbuf, Ccol, Ltmp);
                memcpy(Lbuf, Ltmp, SGM_D);
                unsigned short *Sp = S + ((size_t)y*W + x)*DPAD;
                for (int d = 0; d < SGM_D; d++) Sp[d] = (unsigned short)(Sp[d] + Ltmp[d]);
            }
        }
    } else {
        int y0 = (diry > 0) ? 0 : H-1;
        for (int j = 0; j < H; j++) {
            int y = y0 + diry*j;
            for (int x = 0; x < W; x++) {
                for (int d = 0; d < SGM_D; d++) Ccol[d] = cost_at(cl, cr, W, y, x, d);
                unsigned char *Lp = Lbuf + (size_t)x*DPAD;
                if (j == 0) memcpy(Ltmp, Ccol, SGM_D);
                else step_ref(Lp, Ccol, Ltmp);
                memcpy(Lp, Ltmp, SGM_D);
                unsigned short *Sp = S + ((size_t)y*W + x)*DPAD;
                for (int d = 0; d < SGM_D; d++) Sp[d] = (unsigned short)(Sp[d] + Ltmp[d]);
            }
        }
    }
}
static void build_cost_rows(const unsigned long long *cl, const unsigned long long *cr,
                            unsigned char *CV, int W, int H, int y0, int y1)
{
    for (int y = y0; y < y1; y++)
        for (int x = 0; x < W; x++) {
            unsigned char *c = CV + ((size_t)y*W + x)*DPAD;
            unsigned long long a = cl[(size_t)y*W + x];
            int dmax = (x < SGM_D-1) ? x : SGM_D-1;
            for (int d = 0; d <= dmax; d++)
                c[d] = (unsigned char)__builtin_popcountll(a ^ cr[(size_t)y*W + (x-d)]);
            for (int d = dmax+1; d < SGM_D; d++) c[d] = COST_INVALID;
            for (int d = SGM_D; d < DPAD; d++) c[d] = 255;
        }
}

/* ---- PACKED aggregation: PPV pixels per 128-byte HVX vector ----------------
 * VERTICAL paths pair adjacent COLUMNS (independent, and their cost/L/S records
 * are contiguous, so every access is an aligned 128-byte vector at every PPV).
 * HORIZONTAL paths cannot pair adjacent x -- x+1 depends on x -- so they pair
 * adjacent ROWS; at PPV=4 one row only owns 64 bytes of S per pixel, so two x
 * steps are transposed into one aligned vector per row (hz_store). */
#if VPC == 1
static void path_vert2(const unsigned char *CV, unsigned short *S, int W, int H,
                       int diry, unsigned char *Lrow, int r0, int r1, int first)
{
    const int pf = sgm_pf ? (sgm_pf == 4 ? 0 : sgm_pf) : 1;   /* the S read is the gate */
    const pk_t k = pk_init();
    int y0 = (diry > 0) ? 0 : H-1;
    int xe = r0 + ((r1 - r0) / PPV) * PPV;
    const int PFC = 64;                  /* prefetch chunk, in pixels */
    for (int j = 0; j < H; j++) {
        int y = y0 + diry*j;
        const unsigned char *CVr = CV + (size_t)y*W*DPAD;
        unsigned short     *Sr   = S  + (size_t)y*W*DPAD;
        int c = 0;
        for (int xc = r0; xc < xe; xc += PFC, c++) {
            int xn = xc + PFC; if (xn > xe) xn = xe;
            int xp = xn;
            if (pf && xp < xe) {
                int mode = (pf == 3) ? ((c & 1) ? 2 : 1) : pf;
                if (mode == 1) L2PF(Sr  + (size_t)xp*DPAD, 128, 128, (2*DPAD*PFC)/128);
                else           L2PF(CVr + (size_t)xp*DPAD, 128, 128, (DPAD*PFC)/128);
            }
            int x = xc;
            if (sgm_vch >= 4) for (; x + 4*PPV <= xn; x += 4*PPV) {  /* 4 chains cover the reduce latency */
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
    if (xe < r1) path_leftover(CV, S, W, H, 0, diry, Lrow, xe, r1, first);
}

/* HORIZONTAL sweep, rows [r0,r1) taken PPV at a time. */
static void path_horiz2(const unsigned char *CV, unsigned short *S, int W, int H,
                        int dirx, int r0, int r1, int first)
{
    /* the STORE-only first path has no S read, so its cost-volume reads are the
     * whole story and prefetching them is worth 4.8x */
    const int pf  = sgm_pf ? (sgm_pf == 4 ? 0 : sgm_pf) : (first ? 2 : 1);
    const pk_t k  = pk_init();
    const int PFC = sgm_hpf;
    int x0 = (dirx > 0) ? 0 : W-1;
    int y = r0;
    int ye = r0 + ((r1 - r0) / PPV) * PPV;
    if (W % HB) { path_leftover(CV,S,W,H,dirx,0,0,r0,r1,first); return; }
    for (; y < ye; y += PPV) {
        const unsigned char *A[PPV]; unsigned short *Sr[PPV];
        for (int i = 0; i < PPV; i++) {
            A[i]  = CV + (size_t)(y+i)*W*DPAD;
            Sr[i] = S  + (size_t)(y+i)*W*DPAD;
        }
        HVX_Vector L = k.big;
        HVX_VectorPair wb[HB]; (void)wb;
        int c = 0;
        for (int ic = 0; ic < W; ic += PFC, c++) {
            int inx = ic + PFC; if (inx > W) inx = W;
            if (pf && inx < W) {
                int xp = x0 + dirx*(dirx > 0 ? inx : inx + PFC - 1);
                if (xp >= 0 && xp + PFC <= W) {
                    int mode = (pf == 3) ? ((c & 1) ? 2 : 1) : pf;
                    for (int i = 0; i < PPV; i++) {
                        if (mode == 1) L2PF(Sr[i] + (size_t)xp*DPAD, 128, 128, (2*DPAD*PFC)/128);
                        else           L2PF(A[i]  + (size_t)xp*DPAD, 128, 128, (DPAD*PFC)/128);
                    }
                }
            }
            for (int i = ic; i < inx; i++) {
                int x = x0 + dirx*i;
                HVX_Vector C = hz_cost(A, (size_t)x*DPAD);
                L = (i == 0) ? C : step2(L, C, &k);
#if HB == 1
                HVX_VectorPair w = widen_pair(L);
                hz_store(Sr, (size_t)x*DPAD*2, &w, first);
#else
                wb[x & 1] = widen_pair(L);
                if ((x & 1) == (dirx > 0 ? 1 : 0))
                    hz_store(Sr, (size_t)(x & ~1)*DPAD*2, wb, first);
#endif
            }
        }
    }
    if (ye < r1) path_leftover(CV, S, W, H, dirx, 0, 0, ye, r1, first);
}

/* ---- FUSED L->R + R->L in one row sweep -------------------------------------
 * R->L was the only path still far from the memory ceiling (9.3 GB/s vs ~26
 * for the other three): it is the one that has to READ S back.  Fusing removes
 * that read -- the forward sweep parks its L in a row-local buffer (VTCM), the
 * backward sweep adds the two and does a single STORE into S.
 * Q6_Wh_vadd_VubVub widens and adds the two uint8 columns in one instruction. */
static void path_horiz_fused(const unsigned char *CV, unsigned short *S, int W, int H,
                             int r0, int r1, unsigned char *rowbuf)
{
    const pk_t k = pk_init();
    const int PFC = 64;
    int ye = r0 + ((r1 - r0) / PPV) * PPV;
    if (W % HB) {           /* cannot batch the stores; fall back to two sweeps */
        path_horiz2(CV, S, W, H,  1, r0, r1, 1);
        path_horiz2(CV, S, W, H, -1, r0, r1, 0);
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
                    L2PF(A[i] + (size_t)inx*DPAD, 128, 128, (DPAD*PFC)/128);
            for (int x = ic; x < inx; x++) {
                HVX_Vector C = hz_cost(A, (size_t)x*DPAD);
                L = (x == 0) ? C : step2(L, C, &k);
                *(HVX_Vector *)(rowbuf + (size_t)x*128) = L;
            }
        }
        L = k.big;
        HVX_VectorPair wb[HB]; (void)wb;
        for (int ic = W - 1; ic >= 0; ic -= PFC) {
            int inx = ic - PFC; if (inx < -1) inx = -1;
            if (inx - PFC >= -1)
                for (int i = 0; i < PPV; i++)
                    L2PF(A[i] + (size_t)(inx-PFC+1)*DPAD, 128, 128, (DPAD*PFC)/128);
            for (int x = ic; x > inx; x--) {
                HVX_Vector C = hz_cost(A, (size_t)x*DPAD);
                L = (x == W-1) ? C : step2(L, C, &k);
                HVX_VectorPair w = widen_add_pair(L, *(const HVX_Vector *)(rowbuf + (size_t)x*128));
#if HB == 1
                hz_store(Sr, (size_t)x*DPAD*2, &w, 1);
#else
                wb[x & 1] = w;
                if ((x & 1) == 0) hz_store(Sr, (size_t)x*DPAD*2, wb, 1);
#endif
            }
        }
    }
    if (ye < r1) {                      /* leftover rows: two unpacked sweeps */
        path_horiz2(CV, S, W, H,  1, ye, r1, 1);
        path_horiz2(CV, S, W, H, -1, ye, r1, 0);
    }
}

#endif /* VPC == 1 */

/* ============================================================================
 * VPC == 2 sweeps.  One pixel = two 128-byte L/C vectors and four S vectors,
 * so there is no packing and no leftover: every row and every column is a
 * whole number of pixels at any W and H.  TWO chains, not four -- each pixel
 * already carries two nearly independent vector streams (lo and hi share only
 * the min), and four chains would need 8 live L vectors plus 8 C.
 * ==========================================================================*/
#if VPC == 2
static void path_vert2(const unsigned char *CV, unsigned short *S, int W, int H,
                       int diry, unsigned char *Lrow, int r0, int r1, int first)
{
    const int pf = sgm_pf ? (sgm_pf == 4 ? 0 : sgm_pf) : 1;
    const pk_t k = pk_init();
    int y0 = (diry > 0) ? 0 : H-1;
    const int PFC = 64;
    for (int j = 0; j < H; j++) {
        int y = y0 + diry*j;
        const unsigned char *CVr = CV + (size_t)y*W*DPAD;
        unsigned short     *Sr   = S  + (size_t)y*W*DPAD;
        int c = 0;
        for (int xc = r0; xc < r1; xc += PFC, c++) {
            int xn = xc + PFC; if (xn > r1) xn = r1;
            if (pf && xn < r1) {
                int mode = (pf == 3) ? ((c & 1) ? 2 : 1) : pf;
                if (mode == 1) L2PF(Sr  + (size_t)xn*DPAD, 128, 128, (2*DPAD*PFC)/128);
                else           L2PF(CVr + (size_t)xn*DPAD, 128, 128, (DPAD*PFC)/128);
            }
            int x = xc;
            if (sgm_vch >= 4) for (; x + 2 <= xn; x += 2) {
                const unsigned char *cp = CVr + (size_t)x*DPAD;
                unsigned char *lp = Lrow + (size_t)x*DPAD;
                unsigned short *sp = Sr + (size_t)x*DPAD;
                HVX_Vector C0 = *(const HVX_Vector *)(cp);
                HVX_Vector C1 = *(const HVX_Vector *)(cp + 128);
                HVX_Vector C2 = *(const HVX_Vector *)(cp + 256);
                HVX_Vector C3 = *(const HVX_Vector *)(cp + 384);
                HVX_Vector A0,A1,B0,B1;
                if (j == 0) { A0=C0; A1=C1; B0=C2; B1=C3; }
                else {
                    step2v(*(HVX_Vector *)(lp),       *(HVX_Vector *)(lp + 128), C0, C1, &k, &A0,&A1);
                    step2v(*(HVX_Vector *)(lp + 256), *(HVX_Vector *)(lp + 384), C2, C3, &k, &B0,&B1);
                }
                *(HVX_Vector *)(lp)       = A0;
                *(HVX_Vector *)(lp + 128) = A1;
                *(HVX_Vector *)(lp + 256) = B0;
                *(HVX_Vector *)(lp + 384) = B1;
                acc2v(A0, A1, (HVX_Vector *)(sp),       first);
                acc2v(B0, B1, (HVX_Vector *)(sp + 256), first);
            }
            for (; x < xn; x++) {
                const unsigned char *cp = CVr + (size_t)x*DPAD;
                unsigned char *lp = Lrow + (size_t)x*DPAD;
                HVX_Vector C0 = *(const HVX_Vector *)(cp), C1 = *(const HVX_Vector *)(cp + 128);
                HVX_Vector A0,A1;
                if (j == 0) { A0=C0; A1=C1; }
                else step2v(*(HVX_Vector *)(lp), *(HVX_Vector *)(lp+128), C0, C1, &k, &A0,&A1);
                *(HVX_Vector *)(lp)=A0; *(HVX_Vector *)(lp+128)=A1;
                acc2v(A0, A1, (HVX_Vector *)(Sr + (size_t)x*DPAD), first);
            }
        }
    }
}

static void path_horiz2(const unsigned char *CV, unsigned short *S, int W, int H,
                        int dirx, int r0, int r1, int first)
{
    const int pf  = sgm_pf ? (sgm_pf == 4 ? 0 : sgm_pf) : (first ? 2 : 1);
    const pk_t k  = pk_init();
    const int PFC = sgm_hpf;
    int x0 = (dirx > 0) ? 0 : W-1;
    (void)H;
    for (int y = r0; y < r1; y++) {
        const unsigned char *A  = CV + (size_t)y*W*DPAD;
        unsigned short      *Sr = S  + (size_t)y*W*DPAD;
        HVX_Vector L0 = k.big, L1 = k.big;
        int c = 0;
        for (int ic = 0; ic < W; ic += PFC, c++) {
            int inx = ic + PFC; if (inx > W) inx = W;
            if (pf && inx < W) {
                int xp = x0 + dirx*(dirx > 0 ? inx : inx + PFC - 1);
                if (xp >= 0 && xp + PFC <= W) {
                    int mode = (pf == 3) ? ((c & 1) ? 2 : 1) : pf;
                    if (mode == 1) L2PF(Sr + (size_t)xp*DPAD, 128, 128, (2*DPAD*PFC)/128);
                    else           L2PF(A  + (size_t)xp*DPAD, 128, 128, (DPAD*PFC)/128);
                }
            }
            for (int i = ic; i < inx; i++) {
                int x = x0 + dirx*i;
                const unsigned char *cp = A + (size_t)x*DPAD;
                HVX_Vector C0 = *(const HVX_Vector *)(cp), C1 = *(const HVX_Vector *)(cp+128);
                if (i == 0) { L0 = C0; L1 = C1; }
                else { HVX_Vector n0,n1; step2v(L0,L1,C0,C1,&k,&n0,&n1); L0=n0; L1=n1; }
                acc2v(L0, L1, (HVX_Vector *)(Sr + (size_t)x*DPAD), first);
            }
        }
    }
}

static void path_horiz_fused(const unsigned char *CV, unsigned short *S, int W, int H,
                             int r0, int r1, unsigned char *rowbuf)
{
    const pk_t k = pk_init();
    const int PFC = 64;
    (void)H;
    for (int y = r0; y < r1; y++) {
        const unsigned char *A  = CV + (size_t)y*W*DPAD;
        unsigned short      *Sr = S  + (size_t)y*W*DPAD;
        HVX_Vector L0 = k.big, L1 = k.big;
        for (int ic = 0; ic < W; ic += PFC) {
            int inx = ic + PFC; if (inx > W) inx = W;
            if (inx + PFC <= W) L2PF(A + (size_t)inx*DPAD, 128, 128, (DPAD*PFC)/128);
            for (int x = ic; x < inx; x++) {
                const unsigned char *cp = A + (size_t)x*DPAD;
                HVX_Vector C0 = *(const HVX_Vector *)(cp), C1 = *(const HVX_Vector *)(cp+128);
                if (x == 0) { L0=C0; L1=C1; }
                else { HVX_Vector n0,n1; step2v(L0,L1,C0,C1,&k,&n0,&n1); L0=n0; L1=n1; }
                *(HVX_Vector *)(rowbuf + (size_t)x*256)       = L0;
                *(HVX_Vector *)(rowbuf + (size_t)x*256 + 128) = L1;
            }
        }
        L0 = k.big; L1 = k.big;
        for (int ic = W - 1; ic >= 0; ic -= PFC) {
            int inx = ic - PFC; if (inx < -1) inx = -1;
            if (inx - PFC >= -1) L2PF(A + (size_t)(inx-PFC+1)*DPAD, 128, 128, (DPAD*PFC)/128);
            for (int x = ic; x > inx; x--) {
                const unsigned char *cp = A + (size_t)x*DPAD;
                HVX_Vector C0 = *(const HVX_Vector *)(cp), C1 = *(const HVX_Vector *)(cp+128);
                if (x == W-1) { L0=C0; L1=C1; }
                else { HVX_Vector n0,n1; step2v(L0,L1,C0,C1,&k,&n0,&n1); L0=n0; L1=n1; }
                acc2v_add(L0, L1,
                          *(const HVX_Vector *)(rowbuf + (size_t)x*256),
                          *(const HVX_Vector *)(rowbuf + (size_t)x*256 + 128),
                          (HVX_Vector *)(Sr + (size_t)x*DPAD));
            }
        }
    }
}
#endif /* VPC == 2 */

/* ---- HVX census: 128 pixels per block, bit-exact vs the scalar version ----
 * y-clamping is free (pick the row pointer); only x needs an interior guard,
 * so the first and last CW/2 columns fall back to scalar. The 62 comparisons
 * accumulate into 8 byte-planes which are then interleaved into 8-bytes-per-
 * pixel by three rounds of vshuff at byte/halfword/word granularity -- the
 * transpose 95emulator warned was the hard part. Validated in hexagon-sim
 * against the scalar reference before it ever ran on the board. */
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
    /* One plane at a time with a SINGLE accumulator register.  The previous
     * shape indexed plane[n>>3] inside the 62-neighbour loop, so the array
     * never made it into registers and every neighbour paid a 128-byte spill
     * load + store.  Also Q6_Vb_condacc replaces vmux+vor: one op, not two. */
    for (int p = 0; p < 8; p++) {
        HVX_Vector a = zero;
        int cnt = (p == 7) ? 6 : 8;
        for (int b = 0; b < cnt; b++) {
            int k = p*8 + b;
            const unsigned char *row = im + (size_t)clampi(y + CNS_DY[k], 0, h-1)*w;
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

/* scalar for one x-range of one row (border fallback) */
static void census_span(const unsigned char *im, unsigned long long *out,
                        int w, int h, int y, int x0, int x1)
{
    for (int x = x0; x < x1; x++) {
        unsigned char c = im[(size_t)y*w + x];
        unsigned long long v = 0; int n = 0;
        for (int dy = -(CH/2); dy <= CH/2; dy++)
            for (int dx = -(CW/2); dx <= CW/2; dx++) {
                if (dx == 0 && dy == 0) continue;
                int yy = clampi(y+dy,0,h-1), xx = clampi(x+dx,0,w-1);
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
    /* 1920 is not 4 + 128k, so the plain loop used to stop at x=1796 and hand
     * the last 124 columns of EVERY row to the scalar fallback -- 6.7% of the
     * pixels taking ~600 cycles each, which was about half the census phase.
     * One extra OVERLAPPING block at w-G-128 recomputes some columns (the
     * kernel is idempotent) and leaves only 2*G=8 scalar pixels per row. */
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



/* ---- vectorised build_cost (see cost_group() in sgm_hvx_kernels.h) -------- */
static void build_cost_rows_hvx(const unsigned long long *cl, const unsigned long long *cr,
                                unsigned char *CV, int W, int H, int y0, int y1, int slot)
{
    const pk_t k = pk_init();
    unsigned long long pro = 0;
    const int XS = DPAD;               /* below this, d can exceed x */
    int XV = (XS < W) ? XS : W; XV -= XV % PPV;
#if VPC == 1
    const HVX_Vector dvv = lane_dindex(0), pvv = lane_poffset();
#else
    const HVX_Vector dvlo = lane_dindex(0), dvhi = lane_dindex(1);
    HVX_Vector eplo, ephi; { pk_pad2(&k, &eplo, &ephi); }
#endif
    for (int y = y0; y < y1; y++) {
        const unsigned long long *clr = cl + (size_t)y*W, *crr = cr + (size_t)y*W;
        unsigned char *CVr = CV + (size_t)y*W*DPAD;
        unsigned long long tp0 = HAP_perf_get_pcycles();
        if (sgm_edge) {
#if VPC == 1
            for (int x = 0; x < XV; x += PPV)
                *(HVX_Vector *)(CVr + (size_t)x*DPAD) =
                    cost_group_edge(clr, crr, x, k.pad, dvv, pvv);
#else
            for (int x = 0; x < XV; x++) {
                HVX_Vector clo, chi;
                cost_group2_edge(clr, crr, x, eplo, ephi, dvlo, dvhi, &clo, &chi);
                *(HVX_Vector *)(CVr + (size_t)x*DPAD)       = clo;
                *(HVX_Vector *)(CVr + (size_t)x*DPAD + 128) = chi;
            }
#endif
        }
        for (int x = sgm_edge ? XV : 0; x < XS && x < W; x++) {
            unsigned char *c = CVr + (size_t)x*DPAD;
            unsigned long long a = clr[x];
            int dmax = (x < SGM_D-1) ? x : SGM_D-1;
            for (int d = 0; d <= dmax; d++) c[d] = (unsigned char)__builtin_popcountll(a ^ crr[x-d]);
            for (int d = dmax+1; d < SGM_D; d++) c[d] = COST_INVALID;
            for (int d = SGM_D; d < DPAD; d++) c[d] = 255;
        }
        pro += HAP_perf_get_pcycles() - tp0;
        int x = (XS < W) ? XS : W;
#if VPC == 1
        for (; x + PPV <= W; x += PPV)
            *(HVX_Vector *)(CVr + (size_t)x*DPAD) = cost_group(clr, crr, x, k.pad);
#else
        { HVX_Vector plo, phi; pk_pad2(&k, &plo, &phi);
          for (; x < W; x++) {
              HVX_Vector clo, chi; cost_group2(clr, crr, x, plo, phi, &clo, &chi);
              *(HVX_Vector *)(CVr + (size_t)x*DPAD)       = clo;
              *(HVX_Vector *)(CVr + (size_t)x*DPAD + 128) = chi; } }
#endif
        for (; x < W; x++) {           /* trailing pixels when W % PPV != 0 */
            unsigned char *c = CVr + (size_t)x*DPAD;
            unsigned long long a = clr[x];
            for (int d = 0; d < SGM_D; d++) c[d] = (unsigned char)__builtin_popcountll(a ^ crr[x-d]);
            for (int d = SGM_D; d < DPAD; d++) c[d] = 255;
        }
    }
    if (slot >= 0 && slot < 16) sgm_prolog[slot] = pro;
}

/* ---- multithreading over the NSP's hardware threads ----
 * Every phase splits into disjoint ranges: rows for census / cost / the
 * horizontal paths / argmin, COLUMNS for the vertical paths (a vertical sweep
 * owns a column for its whole length). Workers therefore touch disjoint S and
 * L memory and the result stays bit-exact -- verified against the golden. */
typedef struct {
    worker_synctoken_t *tok;
    const unsigned char *left, *right, *CV;
    const unsigned long long *cl, *cr;
    unsigned long long *clw, *crw;
    unsigned short *S;
    unsigned char *disp, *Lrow;
    int W, H, r0, r1, dirx, diry, use_hvx, first, slot;
} job_t;

static void job_census(void *p){ job_t *j=(job_t*)p;
    if (j->use_hvx) {
        census_rows_hvx(j->left,  j->clw, j->W, j->H, j->r0, j->r1);
        census_rows_hvx(j->right, j->crw, j->W, j->H, j->r0, j->r1);
    } else {
        census_rows(j->left,  j->clw, j->W, j->H, j->r0, j->r1);
        census_rows(j->right, j->crw, j->W, j->H, j->r0, j->r1);
    }
    worker_pool_synctoken_jobdone(j->tok); }

static void job_cost(void *p){ job_t *j=(job_t*)p;
    if (j->use_hvx) build_cost_rows_hvx(j->cl, j->cr, (unsigned char*)j->CV, j->W, j->H, j->r0, j->r1, j->slot);
    else build_cost_rows(j->cl, j->cr, (unsigned char*)j->CV, j->W, j->H, j->r0, j->r1);
    worker_pool_synctoken_jobdone(j->tok); }

static void job_path(void *p){ job_t *j=(job_t*)p;
    if (j->dirx == 2) path_horiz_fused(j->CV, j->S, j->W, j->H, j->r0, j->r1,
                                       sgm_rowbuf_base + (size_t)j->slot*sgm_rowbuf_stride);
    else if (j->diry == 0) path_horiz2(j->CV, j->S, j->W, j->H, j->dirx, j->r0, j->r1, j->first);
    else              path_vert2 (j->CV, j->S, j->W, j->H, j->diry, j->Lrow, j->r0, j->r1, j->first);
    worker_pool_synctoken_jobdone(j->tok); }

static void job_argmin(void *p){ job_t *j=(job_t*)p;
    if (j->use_hvx)
        argmin_hvx(j->S, j->disp, (unsigned long long)j->r0*j->W,
                                  (unsigned long long)j->r1*j->W);
    else for (int y=j->r0; y<j->r1; y++)
        for (int x=0; x<j->W; x++){
            size_t i=(size_t)y*j->W+x;
            const unsigned short *Sp=j->S+i*DPAD;
            unsigned short best=Sp[0]; int bd=0;
            for(int d=1;d<SGM_D;d++) if(Sp[d]<best){best=Sp[d];bd=d;}
            j->disp[i]=(unsigned char)bd;
        }
    worker_pool_synctoken_jobdone(j->tok); }

#define MAXJ 12

/* ---- DSP memory bandwidth probe (use_hvx==9) ------------------------------
 * Settles "is aggregation bandwidth-bound or latency-bound?".  Streams a large
 * region of the ION scratch with plain aligned HVX vector loads / stores. */
typedef struct { worker_synctoken_t *tok; unsigned char *base; size_t n; int mode; } bwjob_t;
static void bw_worker(void *p){
    bwjob_t *j=(bwjob_t*)p;
    HVX_Vector *v=(HVX_Vector*)j->base; size_t nv=j->n/128;
    HVX_Vector acc=Q6_V_vzero();
    if(j->mode==0){ for(size_t i=0;i<nv;i++) acc=Q6_Vub_vmax_VubVub(acc,v[i]);
                    if(Q6_R_vextract_VR(acc,0)==0x7f3a51c2) v[0]=acc; }      /* keep live */
    else if(j->mode==1){ for(size_t i=0;i<nv;i++) v[i]=acc; }
    else { HVX_Vector *d=v+nv; for(size_t i=0;i<nv;i++) d[i]=v[i]; }
    worker_pool_synctoken_jobdone(j->tok);
}
static unsigned long long bw_run(worker_pool_context_t ctx,int nj,unsigned char*base,size_t total,int mode){
    bwjob_t jb[MAXJ]; worker_synctoken_t tok; worker_pool_synctoken_init(&tok,nj);
    size_t per=(total/nj)&~127UL;
    unsigned long long t0=HAP_perf_get_pcycles();
    for(int k=0;k<nj;k++){ jb[k].tok=&tok; jb[k].base=base+(size_t)k*per; jb[k].n=per; jb[k].mode=mode;
        worker_pool_job_t job; job.fptr=bw_worker; job.dptr=&jb[k]; worker_pool_submit(ctx,job); }
    worker_pool_synctoken_wait(&tok);
    return HAP_perf_get_pcycles()-t0;
}

static void run_split(worker_pool_context_t ctx, job_t *tmpl, int nj, int total,
                      worker_callback_t fn, job_t *slots)
{
    worker_synctoken_t tok;
    worker_pool_synctoken_init(&tok, nj);
    int per = (total + nj - 1)/nj;
    per = (per + (PPV-1)) & ~(PPV-1);   /* PPV-aligned: pixel groups never straddle jobs */
    for (int k=0;k<nj;k++){
        slots[k] = *tmpl;
        slots[k].tok = &tok;
        slots[k].slot = k;
        slots[k].r0 = k*per;
        slots[k].r1 = (k+1)*per > total ? total : (k+1)*per;
        if (slots[k].r0 > total) slots[k].r0 = total;
        worker_pool_job_t job; job.fptr = fn; job.dptr = &slots[k];
        worker_pool_submit(ctx, job);
    }
    worker_pool_synctoken_wait(&tok);
}

int sgmdsp_scratch_bytes(remote_handle64 h, int W, int H, uint64 *bytes)
{
    (void)h;
    unsigned long long wh = (unsigned long long)W * H;
    *bytes = wh*8ULL*2ULL                     /* censusL + censusR   */
           + wh*(unsigned long long)DPAD*2ULL /* S (uint16), DPAD per pixel */
           + wh*(unsigned long long)DPAD + 512ULL /* cost volume + front/back pad */
           + (unsigned long long)W*128ULL*VPC     /* L state: DPAD B per column */
           + (unsigned long long)W*128ULL*VPC*12ULL /* per-job row buffers      */
           + 8192ULL;
    return AEE_SUCCESS;
}

int sgmdsp_run(remote_handle64 h,
               const uint8 *left,  int leftLen,
               const uint8 *right, int rightLen,
               int W, int H, int nthreads, int use_hvx_in,
               uint8 *scratch, int scratchLen,
               uint8 *disp, int dispLen,
               uint64 *cycles,
               uint64 *phases, int phasesLen)
{
    (void)h;
    sgm_pf  = (use_hvx_in >> 4) & 3;
    sgm_vch = ((use_hvx_in >> 12) & 7) ? 1 : 4;
    sgm_hpf = ((use_hvx_in >> 16) & 7) ? (64 << ((use_hvx_in >> 16) & 7)) : 64;
    sgm_fuse = ((use_hvx_in >> 20) & 1) ? 0 : 1;
    sgm_vtcm = ((use_hvx_in >> 21) & 1) ? 0 : 1;
    sgm_edge = ((use_hvx_in >> 22) & 1) ? 0 : 1;
    int use_hvx = use_hvx_in & 0xf;
    unsigned long long wh = (unsigned long long)W * H;
    if (leftLen  < (int)wh || rightLen < (int)wh || dispLen < (int)wh) return AEE_EBADPARM;

    uint64 need = 0; sgmdsp_scratch_bytes(h, W, H, &need);
    if ((unsigned long long)scratchLen < need) return AEE_EBADPARM;

    sgm_leftover = 0;
    memset(sgm_prolog, 0, sizeof sgm_prolog);
    unsigned long long t0 = HAP_perf_get_pcycles();

    unsigned char *p = scratch;
    /* keep every vector buffer 128-byte aligned */
    unsigned long long *cl = (unsigned long long *)p; p += wh*8;
    unsigned long long *cr = (unsigned long long *)p; p += wh*8;
    unsigned short *S = (unsigned short *)p;          p += wh*(unsigned long long)DPAD*2;
    p += 128;                                          /* front pad for path_horiz2 */
    unsigned char *CV = p;                            p += wh*(unsigned long long)DPAD + 128;
    p = (unsigned char *)(((unsigned long)p + 127) & ~127UL);
    unsigned char *Lbuf = p;

    int nj  = nthreads & 0xff;            /* scalar-phase job count */
    int njv = (nthreads >> 8) & 0xff;      /* HVX-phase job count (0 -> same) */
    if (nj < 1) nj = 1; if (nj > MAXJ) nj = MAXJ;
    if (njv < 1) njv = nj; if (njv > MAXJ) njv = MAXJ;
    worker_pool_context_t ctx = NULL;
    if (nj > 1 && worker_pool_init(&ctx) != 0) { ctx = NULL; nj = 1; }
    job_t slots[MAXJ], t; memset(&t,0,sizeof t);
    t.left=left; t.right=right; t.clw=cl; t.crw=cr; t.cl=cl; t.cr=cr;
    t.CV=CV; t.S=S; t.disp=disp; t.Lrow=Lbuf; t.W=W; t.H=H; t.use_hvx=use_hvx;

    if (use_hvx == 9) {                     /* bandwidth probe, NOT a golden run */
        size_t region = (size_t)(wh*(unsigned long long)DPAD*2) / 2;   /* x2 for copy */
        if (ctx) {
            phases[0]=bw_run(ctx,nj,(unsigned char*)S,region,0);
            phases[1]=bw_run(ctx,nj,(unsigned char*)S,region,1);
            phases[2]=bw_run(ctx,nj,(unsigned char*)S,region,2);
        }
        phases[3]=region; phases[4]=num_workers; phases[5]=num_hvx128_contexts;
        if (ctx) worker_pool_deinit(&ctx);
        *cycles = HAP_perf_get_pcycles() - t0;
        return AEE_SUCCESS;
    }
    /* VTCM for the fused row buffers.  The measured capability query says this
     * part has 8 MB of VTCM in one page; the fused horizontal sweep needs
     * W*128 bytes per worker (983 KB at 1920 x 4 workers), which the L2 cannot
     * hold against a streaming cost volume and a streaming S. */
    compute_res_attr_t vattr;
    unsigned vctx = 0;
    sgm_rowbuf_stride = (unsigned long)W*128*VPC;
    sgm_rowbuf_base   = Lbuf + sgm_rowbuf_stride;     /* scratch fallback */
    if (sgm_vtcm && use_hvx && compute_resource_attr_init &&
        compute_resource_attr_set_vtcm_param && compute_resource_acquire &&
        compute_resource_attr_get_vtcm_ptr &&
        compute_resource_attr_init(&vattr) == 0) {
        unsigned vsz = (unsigned)(sgm_rowbuf_stride * (unsigned)(njv + 1));
        if (compute_resource_attr_set_serialize) compute_resource_attr_set_serialize(&vattr, 0);
        compute_resource_attr_set_vtcm_param(&vattr, vsz, 1);
        vctx = compute_resource_acquire(&vattr, 100000);
        if (vctx) {
            void *vp = compute_resource_attr_get_vtcm_ptr(&vattr);
            if (vp) {
                /* [0, W*128) = the vertical paths' L state (one 64-byte record
                 * per column, re-read every row); the rest = one W*128 row
                 * buffer per worker for the fused horizontal sweep. */
                t.Lrow = (unsigned char *)vp;
                sgm_rowbuf_base = (unsigned char *)vp + sgm_rowbuf_stride;
            }
            else { compute_resource_release(vctx); vctx = 0; }
        }
    }
    if (phasesLen >= 12) phases[11] = vctx ? 1 : 0;
    if (phasesLen >= 14) { phases[12] = SGM_D; phases[13] = 0; }

    unsigned long long tA = HAP_perf_get_pcycles();
    if (nj > 1) run_split(ctx,&t,use_hvx?njv:nj,H,job_census,slots);
    else if (use_hvx) { census_rows_hvx(left,cl,W,H,0,H); census_rows_hvx(right,cr,W,H,0,H); }
    else { census_rows(left,cl,W,H,0,H); census_rows(right,cr,W,H,0,H); }
    unsigned long long tB = HAP_perf_get_pcycles();
    /* memset eliminated: the first path STORES into S instead of accumulating.
     * The 265 MB memset was single-threaded and lived inside the "cost" phase. */
    unsigned long long tM = HAP_perf_get_pcycles();

    unsigned long long tC = tB, tD = tB, tP[4]={0,0,0,0};
    if (use_hvx) {
        if (nj > 1) run_split(ctx,&t,nj,H,job_cost,slots);
        else build_cost_rows_hvx(cl,cr,CV,W,H,0,H,0);
        tC = HAP_perf_get_pcycles();
        if (nj > 1) {
            if (sgm_fuse) {
                t.first=1; t.diry=0; t.dirx= 2; run_split(ctx,&t,njv,H,job_path,slots);
                tP[0]=HAP_perf_get_pcycles(); tP[1]=tP[0];
                t.first=0;      /* the fused sweep already STORED both horizontal paths */
            } else {
                t.first=1; t.diry=0; t.dirx= 1; run_split(ctx,&t,njv,H,job_path,slots);
                tP[0]=HAP_perf_get_pcycles();
                t.first=0; t.diry=0; t.dirx=-1; run_split(ctx,&t,njv,H,job_path,slots);
                tP[1]=HAP_perf_get_pcycles();
            }
            t.dirx=0; t.diry= 1; run_split(ctx,&t,njv,W,job_path,slots);
            tP[2]=HAP_perf_get_pcycles();
            t.dirx=0; t.diry=-1; run_split(ctx,&t,njv,W,job_path,slots);
            tP[3]=HAP_perf_get_pcycles();
        } else {
            path_horiz2(CV,S,W,H, 1,0,H,1);      tP[0]=HAP_perf_get_pcycles();
            path_horiz2(CV,S,W,H,-1,0,H,0);      tP[1]=HAP_perf_get_pcycles();
            path_vert2 (CV,S,W,H, 1,Lbuf,0,W,0); tP[2]=HAP_perf_get_pcycles();
            path_vert2 (CV,S,W,H,-1,Lbuf,0,W,0); tP[3]=HAP_perf_get_pcycles();
        }
        tD = HAP_perf_get_pcycles();
    } else {
        memset(S, 0, (size_t)(wh*(unsigned long long)DPAD*2));
        path(cl, cr, S, W, H,  1, 0, Lbuf);
        path(cl, cr, S, W, H, -1, 0, Lbuf);
        path(cl, cr, S, W, H, 0,  1, Lbuf);
        path(cl, cr, S, W, H, 0, -1, Lbuf);
    }

    unsigned long long tE = HAP_perf_get_pcycles();
    if (phasesLen >= 4) {
        phases[0] = tB - tA;      /* census      */
        phases[1] = tC - tB;      /* build_cost  */
        phases[2] = tD - tC;      /* aggregate   */
        phases[3] = 0;            /* argmin, filled below */
    }
    if (phasesLen >= 6) {
        phases[4] = num_workers;
        phases[5] = num_hvx128_contexts;
    }
    if (phasesLen >= 11) {
        phases[6]  = tM - tB;            /* (memset slot, now ~0) */
        phases[7]  = tP[0] ? tP[0]-tC : 0;
        phases[8]  = tP[1] ? tP[1]-tP[0] : 0;
        phases[9]  = tP[2] ? tP[2]-tP[1] : 0;
        phases[10] = tP[3] ? tP[3]-tP[2] : 0;
    }
    if (nj > 1) run_split(ctx,&t,nj,H,job_argmin,slots);
    else if (use_hvx) argmin_hvx(S, disp, 0, wh);
    else for (unsigned long long i = 0; i < wh; i++) {
        const unsigned short *Sp = S + i*DPAD;
        unsigned short best = Sp[0]; int bd = 0;
        for (int d = 1; d < SGM_D; d++) if (Sp[d] < best) { best = Sp[d]; bd = d; }
        disp[i] = (unsigned char)bd;      /* lowest d wins ties: strict < */
    }
    if (ctx) worker_pool_deinit(&ctx);
    if (vctx && compute_resource_release) compute_resource_release(vctx);

    if (phasesLen >= 14) phases[13] = sgm_leftover;
    if (phasesLen >= 15) { unsigned long long mx = 0;
        for (int i = 0; i < 16; i++) if (sgm_prolog[i] > mx) mx = sgm_prolog[i];
        phases[14] = mx; }
    if (phasesLen >= 4) phases[3] = HAP_perf_get_pcycles() - tE;
    *cycles = HAP_perf_get_pcycles() - t0;
    return AEE_SUCCESS;
}

int sgmdsp_open(const char *uri, remote_handle64 *h)  { (void)uri; *h = 0xdeadc0de; return AEE_SUCCESS; }
int sgmdsp_close(remote_handle64 h)                   { (void)h; return AEE_SUCCESS; }
