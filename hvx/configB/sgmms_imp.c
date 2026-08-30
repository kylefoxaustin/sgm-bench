/* sgmms_imp.c — Configuration B (multiscale SGM) on the Hexagon NSP.
 *
 * Orchestration only: every compute kernel lives in sgmms_kernels.h, which
 * sim/msb.c validates byte-for-byte against a scalar port of the oracle
 * (sgm-bench/multiscale/sgm_ms_ref.c) before anything runs on the board.
 *
 * Parallel decomposition (all worker-disjoint in S, hence bit-exact):
 *   sobel / census / cost / horizontal paths / argmin : by ROWS
 *   vertical paths                                    : by COLUMNS
 *   diagonal paths : by DIAGONAL CHAINS (a chain is one dependence line;
 *     a worker owns a band of chains, its column window slides one pixel per
 *     row, bands partition every row's pixels -> no synchronisation at all).
 *     Bands are cut at equal PIXEL counts, not equal chain counts (edge
 *     chains are short; equal-chain cuts cost ~1.4x on the diagonals).
 */
#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_perf.h"
#include "sgmms.h"
#include "AEEStdDef.h"
#include <string.h>
#include <stdlib.h>
#include "worker_pool.h"
#include "HAP_compute_res.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

/* l2fetch: one command at a time; the SDK's l2fetch() helper is a stub. */
static inline void l2pf(const void *a, unsigned stride, unsigned width, unsigned height)
{
    unsigned long long param = ((unsigned long long)stride << 32)
                             | ((unsigned long long)width << 16) | height;
    __asm__ __volatile__ ("l2fetch(%0,%1)" : : "r"(a), "r"(param));
}
#define SGM_L2PF(a,s,w,h) l2pf((const void*)(a),(s),(w),(h))
#define SGM_PCYC() HAP_perf_get_pcycles()

#include "sgmms_kernels.h"

#define MAXJ 12

/* ---- job plumbing ---- */
typedef struct {
    worker_synctoken_t *tok;
    const unsigned char *imL, *imR;      /* stage input images            */
    unsigned char *sobL, *sobR;          /* sobel outputs                 */
    unsigned long long *clw, *crw;       /* census outputs                */
    const unsigned char *CV;             /* cost volume of this stage     */
    const unsigned char *C1;             /* half-res cost volume (merge)  */
    unsigned short *S;
    unsigned char *disp;
    unsigned char *fullim;               /* full-res image (downsample)   */
    short *Tbase;                        /* sobel T buffers, per worker   */
    size_t Tstride;
    unsigned char *dgbase;               /* diag L buffers, per worker    */
    size_t dgstride;
    unsigned char *Lrow;                 /* vertical L state (shared)     */
    unsigned char *rowbuf;               /* fused-horizontal row buffers  */
    size_t rbstride;
    int W, H, hw, ow, r0, r1, dirx, diry, use_hvx, first, slot, p1;
} job_t;

static void job_half(void *p){ job_t *j=(job_t*)p;
    if (j->use_hvx) {
        half_rows_hvx(j->fullim, j->sobL, j->W, j->H, j->r0, j->r1);
    } else {
        half_rows(j->fullim, j->sobL, j->W, j->H, j->r0, j->r1);
    }
    worker_pool_synctoken_jobdone(j->tok); }

static void job_sobel(void *p){ job_t *j=(job_t*)p;
    if (j->use_hvx) {
        short *T = (short *)((unsigned char *)j->Tbase + (size_t)j->slot*j->Tstride) + 64;
        sobel_rows_hvx(j->imL, j->sobL, j->W, j->H, j->r0, j->r1, T);
        sobel_rows_hvx(j->imR, j->sobR, j->W, j->H, j->r0, j->r1, T);
    } else {
        sobel_rows(j->imL, j->sobL, j->W, j->H, j->r0, j->r1);
        sobel_rows(j->imR, j->sobR, j->W, j->H, j->r0, j->r1);
    }
    worker_pool_synctoken_jobdone(j->tok); }

static void job_census(void *p){ job_t *j=(job_t*)p;
    if (j->use_hvx) {
        census_rows_hvx(j->sobL, j->clw, j->W, j->H, j->r0, j->r1);
        census_rows_hvx(j->sobR, j->crw, j->W, j->H, j->r0, j->r1);
    } else {
        census_rows(j->sobL, j->clw, j->W, j->H, j->r0, j->r1);
        census_rows(j->sobR, j->crw, j->W, j->H, j->r0, j->r1);
    }
    worker_pool_synctoken_jobdone(j->tok); }

static void job_cost(void *p){ job_t *j=(job_t*)p;
    if (j->use_hvx)
        build_cost_rows_hvx_b(j->clw, j->crw, (unsigned char*)j->CV, j->W, j->H,
                              j->r0, j->r1, j->slot, j->C1, j->hw);
    else
        build_cost_rows_b(j->clw, j->crw, (unsigned char*)j->CV, j->W, j->H,
                          j->r0, j->r1, j->C1, j->hw);
    worker_pool_synctoken_jobdone(j->tok); }

static void job_path(void *p){ job_t *j=(job_t*)p;
    if (j->dirx == 2)
        path_horiz_fused_b(j->CV, j->S, j->W, j->H, j->r0, j->r1,
                           j->rowbuf + (size_t)j->slot*j->rbstride, j->p1);
    else if (j->diry == 0)
        path_horiz2_b(j->CV, j->S, j->W, j->H, j->dirx, j->r0, j->r1, j->first, j->p1);
    else if (j->dirx == 0)
        path_vert2_b(j->CV, j->S, j->W, j->H, j->diry, j->Lrow, j->r0, j->r1, j->first, j->p1);
    else {
        unsigned char *dg = j->dgbase + (size_t)j->slot*j->dgstride;
        /* dgstride/64 slots >= 2(W+2) >= W+H-1+2 chains at any band cut */
        path_diag_b(j->CV, j->S, j->W, j->H, j->dirx, j->diry, j->p1,
                    dg, j->r0, j->r1);
    }
    worker_pool_synctoken_jobdone(j->tok); }

static void job_argmin(void *p){ job_t *j=(job_t*)p;
    if (j->diry == 0)
        argmin_b_range(j->S, j->disp, (unsigned long long)j->r0*j->W,
                                      (unsigned long long)j->r1*j->W);
    else
        argmin_b_dec(j->S, j->disp, j->W, j->ow, j->r0, j->r1);
    worker_pool_synctoken_jobdone(j->tok); }

static void job_argmin_scalar(void *p){ job_t *j=(job_t*)p;
    for (unsigned long long i=(unsigned long long)j->r0*j->W;
         i<(unsigned long long)j->r1*j->W; i++){
        const unsigned short *Sp=j->S+i*DPAD;
        unsigned short best=Sp[0]; int bd=0;
        for(int d=1;d<SGM_D;d++) if(Sp[d]<best){best=Sp[d];bd=d;}
        j->disp[i]=(unsigned char)bd;
    }
    worker_pool_synctoken_jobdone(j->tok); }

static void run_split(worker_pool_context_t ctx, job_t *tmpl, int nj, int total,
                      worker_callback_t fn, job_t *slots)
{
    worker_synctoken_t tok;
    worker_pool_synctoken_init(&tok, nj);
    int per = (total + nj - 1)/nj;
    per = (per + (PPV-1)) & ~(PPV-1);
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
/* diagonal paths: explicit chain-band cuts instead of an even split */
static void run_diag(worker_pool_context_t ctx, job_t *tmpl, int nj,
                     int dirx, int diry, job_t *slots)
{
    int cut[MAXJ+1];
    diag_split(tmpl->W, tmpl->H, dirx, nj, cut);
    worker_synctoken_t tok;
    worker_pool_synctoken_init(&tok, nj);
    for (int k=0;k<nj;k++){
        slots[k] = *tmpl;
        slots[k].tok = &tok; slots[k].slot = k;
        slots[k].dirx = dirx; slots[k].diry = diry; slots[k].p1 = 10;
        slots[k].r0 = cut[k]; slots[k].r1 = cut[k+1];
        worker_pool_job_t job; job.fptr = job_path; job.dptr = &slots[k];
        worker_pool_submit(ctx, job);
    }
    worker_pool_synctoken_wait(&tok);
}

int sgmms_scratch_bytes(remote_handle64 h, int W, int H, uint64 *bytes)
{
    (void)h;
    unsigned long long wh  = (unsigned long long)W * H;
    unsigned long long whh = wh / 4;
    unsigned long long dg  = 2ULL * (((unsigned long long)(W+2)*64 + 127) & ~127ULL);
    *bytes = wh*8ULL*2ULL                     /* censusL + censusR (full res)   */
           + wh*(unsigned long long)DPAD*2ULL /* S (uint16)                     */
           + wh*(unsigned long long)DPAD + 512ULL /* CV2 + front/back pad       */
           + whh*(unsigned long long)DPAD + 256ULL /* C1 (persists into merge)  */
           + wh*2ULL                          /* sobel L + R (stage reuse)      */
           + whh*2ULL                         /* half-res image pair            */
           + whh + 128ULL                     /* stage-1 argmin output          */
           + (unsigned long long)MAXJ*((W+16)*2 + 512) /* sobel T buffers       */
           + (unsigned long long)MAXJ*dg      /* diag L double buffers          */
           + (unsigned long long)W*128ULL     /* vertical L state (fallback)    */
           + (unsigned long long)W*128ULL*MAXJ/* fused row buffers (fallback)   */
           + 16384ULL;
    return AEE_SUCCESS;
}

int sgmms_run(remote_handle64 h,
              const uint8 *left,  int leftLen,
              const uint8 *right, int rightLen,
              int W, int H, int nthreads, int use_hvx,
              uint8 *scratch, int scratchLen,
              uint8 *disp, int dispLen,
              uint64 *cycles,
              uint64 *phases, int phasesLen)
{
    (void)h;
    int hw = W/2, hh = H/2, ow = W/2, oh = H/2;
    unsigned long long wh = (unsigned long long)W * H;
    unsigned long long whh = (unsigned long long)hw * hh;
    if (leftLen < (int)wh || rightLen < (int)wh || dispLen < (int)((unsigned long long)ow*oh))
        return AEE_EBADPARM;
    uint64 need = 0; sgmms_scratch_bytes(h, W, H, &need);
    if ((unsigned long long)scratchLen < need) return AEE_EBADPARM;
    if (phasesLen < 24) return AEE_EBADPARM;

    sgm_leftover = 0;
    memset(sgm_prolog, 0, sizeof sgm_prolog);
    memset(phases, 0, 24*sizeof(uint64));
    unsigned long long t0 = HAP_perf_get_pcycles();

    /* ---- scratch carve (all vector buffers 128-byte aligned) ---- */
    unsigned char *p = scratch;
    unsigned long long *cl = (unsigned long long *)p; p += wh*8;
    unsigned long long *cr = (unsigned long long *)p; p += wh*8;
    unsigned short *S = (unsigned short *)p;          p += wh*(unsigned long long)DPAD*2;
    p += 128;                                  /* front pad for hz_cost      */
    unsigned char *CV = p;                     p += wh*(unsigned long long)DPAD + 128;
    p = (unsigned char *)(((unsigned long)p + 127) & ~127UL);
    unsigned char *C1 = p;                     p += whh*(unsigned long long)DPAD + 128;
    p = (unsigned char *)(((unsigned long)p + 127) & ~127UL);
    unsigned char *sobL = p;                   p += wh;
    unsigned char *sobR = p;                   p += wh;
    unsigned char *imLh = p;                   p += whh;
    unsigned char *imRh = p;                   p += whh;
    unsigned char *dispH = p;                  p += whh + 128;
    p = (unsigned char *)(((unsigned long)p + 127) & ~127UL);
    short *Tbase = (short *)p;
    size_t Tstride = (((size_t)(W+16)*2 + 512) + 127) & ~(size_t)127;
    p += (size_t)MAXJ * Tstride;
    size_t dgstride = 2ULL * ((((size_t)(W+2))*64 + 127) & ~(size_t)127);
    unsigned char *dgbase = p;                 p += (size_t)MAXJ * dgstride;
    unsigned char *Lrow_dr = p;                p += (size_t)W*128;
    unsigned char *rowbuf_dr = p;              /* MAXJ * W*128 to the end */

    int nj = nthreads & 0xff;
    if (nj < 1) nj = 1; if (nj > MAXJ) nj = MAXJ;
    worker_pool_context_t ctx = NULL;
    if (nj > 1 && worker_pool_init(&ctx) != 0) { ctx = NULL; nj = 1; }

    /* ---- VTCM for the hot row-state buffers ---- */
    compute_res_attr_t vattr;
    unsigned vctx = 0;
    size_t rbstride = (size_t)W*128;
    unsigned char *Lrow = Lrow_dr, *rowbuf = rowbuf_dr, *dgb = dgbase;
    if (use_hvx && compute_resource_attr_init &&
        compute_resource_attr_set_vtcm_param && compute_resource_acquire &&
        compute_resource_attr_get_vtcm_ptr &&
        compute_resource_attr_init(&vattr) == 0) {
        unsigned vsz = (unsigned)(rbstride*(nj+1) + (size_t)nj*dgstride);
        if (compute_resource_attr_set_serialize) compute_resource_attr_set_serialize(&vattr, 0);
        compute_resource_attr_set_vtcm_param(&vattr, vsz, 1);
        vctx = compute_resource_acquire(&vattr, 100000);
        if (vctx) {
            void *vp = compute_resource_attr_get_vtcm_ptr(&vattr);
            if (vp) {
                Lrow   = (unsigned char *)vp;              /* W*128           */
                rowbuf = Lrow + rbstride;                  /* nj * W*128      */
                dgb    = rowbuf + rbstride*nj;             /* nj * dgstride   */
            } else { compute_resource_release(vctx); vctx = 0; }
        }
    }

    job_t slots[MAXJ], t; memset(&t, 0, sizeof t);
    t.S = S; t.use_hvx = use_hvx; t.hw = hw; t.ow = ow;
    t.Tbase = Tbase; t.Tstride = Tstride;
    t.dgbase = dgb; t.dgstride = dgstride;
    t.Lrow = Lrow; t.rowbuf = rowbuf; t.rbstride = rbstride;

    unsigned long long tp[12]; int np = 0;
    tp[np++] = HAP_perf_get_pcycles();

    /* ==== downsample (full -> half) ==== */
    t.fullim = (unsigned char *)left;  t.sobL = imLh; t.W = W; t.H = H;
    run_split(ctx, &t, nj, hh, job_half, slots);
    t.fullim = (unsigned char *)right; t.sobL = imRh;
    run_split(ctx, &t, nj, hh, job_half, slots);
    tp[np++] = HAP_perf_get_pcycles();

    for (int stage = 0; stage < 2; stage++) {
        int w = stage ? W : hw, hgt = stage ? H : hh;
        t.W = w; t.H = hgt;
        t.imL = stage ? left : imLh; t.imR = stage ? right : imRh;
        t.sobL = sobL; t.sobR = sobR;
        t.clw = cl; t.crw = cr;
        t.CV = stage ? CV : C1;
        t.C1 = stage ? C1 : NULL;

        run_split(ctx, &t, nj, hgt, job_sobel, slots);
        tp[np++] = HAP_perf_get_pcycles();
        run_split(ctx, &t, nj, hgt, job_census, slots);
        tp[np++] = HAP_perf_get_pcycles();
        run_split(ctx, &t, nj, hgt, job_cost, slots);
        tp[np++] = HAP_perf_get_pcycles();

        if (use_hvx) {
            t.first = 1; t.diry = 0; t.dirx = 2; t.p1 = 40;      /* fused L->R + R->L */
            run_split(ctx, &t, nj, hgt, job_path, slots);
            t.first = 0;
            t.dirx = 0; t.diry =  1; t.p1 = 20;
            run_split(ctx, &t, nj, w, job_path, slots);
            t.dirx = 0; t.diry = -1; t.p1 = 20;
            run_split(ctx, &t, nj, w, job_path, slots);
            unsigned long long dv0 = HAP_perf_get_pcycles();
            run_diag(ctx, &t, nj,  1,  1, slots);
            run_diag(ctx, &t, nj, -1, -1, slots);
            run_diag(ctx, &t, nj,  1, -1, slots);
            run_diag(ctx, &t, nj, -1,  1, slots);
            if (stage) phases[22] = HAP_perf_get_pcycles() - dv0;
        } else {
            /* scalar cross-check: single-threaded oracle port */
            memset(S, 0, (size_t)w*hgt*DPAD*2);
            aggregate_scalar_b((const unsigned char *)t.CV, S, w, hgt,
                               dgb, dgb + dgstride/2);
        }
        tp[np++] = HAP_perf_get_pcycles();

        t.disp = stage ? disp : dispH;
        if (use_hvx) {
            if (!stage) { t.diry = 0; run_split(ctx, &t, nj, hgt, job_argmin, slots); }
            else        { t.diry = 1; run_split(ctx, &t, nj, oh, job_argmin, slots); }
        } else {
            if (!stage) run_split(ctx, &t, nj, hgt, job_argmin_scalar, slots);
            else {      /* scalar decimated argmin */
                for (int y = 0; y < oh; y++) for (int x = 0; x < ow; x++) {
                    const unsigned short *Sp = S + ((size_t)(2*y)*w + 2*x)*DPAD;
                    unsigned short best = Sp[0]; int bd = 0;
                    for (int d = 1; d < SGM_D; d++) if (Sp[d] < best){best=Sp[d];bd=d;}
                    disp[(size_t)y*ow + x] = (unsigned char)bd;
                }
            }
        }
        tp[np++] = HAP_perf_get_pcycles();
    }

    if (ctx) worker_pool_deinit(&ctx);
    if (vctx && compute_resource_release) compute_resource_release(vctx);

    /* phases: 0 half | 1 s1sobel 2 s1census 3 s1cost 4 s1agg 5 s1argmin
     *         6 s2sobel 7 s2census 8 s2cost+merge 9 s2agg 10 s2argmin      */
    for (int i = 0; i < 11 && i+1 < np; i++) phases[i] = tp[i+1] - tp[i];
    phases[11] = vctx ? 1 : 0;
    phases[12] = SGM_D;
    phases[13] = sgm_leftover;
    phases[14] = num_workers;
    phases[15] = num_hvx128_contexts;
    { unsigned long long mx = 0;
      for (int i = 0; i < 16; i++) if (sgm_prolog[i] > mx) mx = sgm_prolog[i];
      phases[16] = mx; }
    *cycles = HAP_perf_get_pcycles() - t0;
    return AEE_SUCCESS;
}

int sgmms_open(const char *uri, remote_handle64 *hh) { (void)uri; *hh = 0xdeadc0de; return AEE_SUCCESS; }
int sgmms_close(remote_handle64 hh)                  { (void)hh; return AEE_SUCCESS; }
