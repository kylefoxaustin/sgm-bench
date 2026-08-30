/* multiscale/sgm_ms_neon.c — Configuration B for the Arm cores (NEON + OpenMP).
 *
 * The tuned CPU implementation of Configuration B, at parity with what
 * a55/sgm_a55.c is for the primary configuration. Reuses the proven pieces —
 * census_neon (identical 9x7/62-bit layout) and the ham16 popcount tree — and
 * adds what B needs: the SobelX prefilter, two full scales with materialised
 * and AVERAGED cost volumes, EIGHT paths with per-direction P1, P2=200 with
 * SATURATING arithmetic (the primary impl's unclamped final add relies on
 * cost+P2 <= 255, which does not hold at P2=200), and a half-resolution
 * output. S is uint16: eight paths of <=255 can reach 2040.
 *
 * Parallelism, the same output-neutral shape as the oracle's: horizontal
 * paths parallel over rows (each row an independent recurrence); vertical and
 * diagonal paths parallel over columns within a row (prev comes from the
 * previous row's buffer). Bit-exact against multiscale/sgm_ms_ref.c.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/sgm.h"
#include "../common/census_neon.h"
#include "../common/hamming_neon.h"
#include <arm_neon.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define D 64
#define DREG 4                    /* 64 disparities = 4 uint8x16 */
#define P2V 200
#define CINVAL 63

static inline int clampi(int v,int lo,int hi){return v<lo?lo:(v>hi?hi:v);}

static void sobelx(const uint8_t *im, uint8_t *out, int W, int H){
    #pragma omp parallel for schedule(static)
    for (int y=0;y<H;y++) for (int x=0;x<W;x++){
        int xm=clampi(x-1,0,W-1),xp=clampi(x+1,0,W-1);
        int ym=clampi(y-1,0,H-1),yp=clampi(y+1,0,H-1);
        int gx=-im[ym*W+xm]+im[ym*W+xp]-2*im[y*W+xm]+2*im[y*W+xp]-im[yp*W+xm]+im[yp*W+xp];
        out[y*W+x]=(uint8_t)clampi(128+(gx>>3),0,255);
    }
}
static void half(const uint8_t *im,uint8_t *o,int W,int H){
    int w=W/2,h=H/2;
    #pragma omp parallel for schedule(static)
    for (int y=0;y<h;y++) for (int x=0;x<w;x++)
        o[y*w+x]=(uint8_t)((im[(2*y)*W+2*x]+im[(2*y)*W+2*x+1]
                           +im[(2*y+1)*W+2*x]+im[(2*y+1)*W+2*x+1]+2)>>2);
}
/* materialised cost volume via the ham16 tree; interior vectorised, edges scalar */
static void costvol(const uint64_t *cl,const uint64_t *cr,uint8_t *C,int W,int H){
    #pragma omp parallel for schedule(static)
    for (int y=0;y<H;y++){
        const uint64_t *rl=cl+(size_t)y*W, *rr=cr+(size_t)y*W;
        for (int x=0;x<W;x++){
            uint8_t *c=C+((size_t)y*W+x)*D;
            if (x>=D+14){                       /* whole 16-lane groups valid */
                for (int d0=0;d0<D;d0+=16){
                    uint8x16_t v=ham16_neon(rl,rr,x,d0);
                    vst1q_u8(c+d0,v);
                }
            } else {
                for (int d=0;d<D;d++){ int xr=x-d;
                    c[d]=(xr<0)?CINVAL:(uint8_t)__builtin_popcountll(rl[x]^rr[xr]); }
            }
        }
    }
}
static void costavg(uint8_t *C2,const uint8_t *C1,int W,int H){
    int hw=W/2;
    #pragma omp parallel for schedule(static)
    for (int y=0;y<H;y++) for (int x=0;x<W;x++){
        uint8_t *c2=C2+((size_t)y*W+x)*D;
        const uint8_t *c1=C1+((size_t)(y/2)*hw+(x/2))*D;
        for (int d=0;d<D;d+=2){                 /* pairs share c1[d/2] */
            uint8_t a=c1[d>>1];
            c2[d]  =(uint8_t)((c2[d]  +a+1)>>1);
            c2[d+1]=(uint8_t)((c2[d+1]+a+1)>>1);
        }
    }
}
/* one NEON recurrence step, 64 disparities in 4 regs, SATURATING, per-pass P1 */
static inline void step_neon(const uint8x16_t *pv,const uint8_t *cost,
                             uint8x16_t *cur,uint8x16_t p1v){
    uint8x16_t mnv=vminq_u8(vminq_u8(pv[0],pv[1]),vminq_u8(pv[2],pv[3]));
    uint8_t mn=vminvq_u8(mnv);
    uint8x16_t mnq=vdupq_n_u8(mn);
    uint8x16_t e=vqaddq_u8(mnq,vdupq_n_u8(P2V));
    uint8x16_t big=vdupq_n_u8(255);
    for (int r=0;r<DREG;r++){
        uint8x16_t lo=(r==0)?vextq_u8(big,pv[0],15):vextq_u8(pv[r-1],pv[r],15);
        uint8x16_t hi=(r==DREG-1)?vextq_u8(pv[r],big,1):vextq_u8(pv[r],pv[r+1],1);
        uint8x16_t m=pv[r];
        m=vminq_u8(m,vqaddq_u8(lo,p1v));
        m=vminq_u8(m,vqaddq_u8(hi,p1v));
        m=vminq_u8(m,e);
        uint8x16_t c=vld1q_u8(cost+r*16);
        cur[r]=vqaddq_u8(c,vsubq_u8(m,mnq));    /* SATURATING: P2=200 overflows */
    }
}
static inline void s_add(uint16_t *s,const uint8x16_t *L){
    for (int r=0;r<DREG;r++){
        uint16x8_t a=vld1q_u16(s+r*16), b=vld1q_u16(s+r*16+8);
        a=vaddw_u8(a,vget_low_u8(L[r])); b=vaddw_u8(b,vget_high_u8(L[r]));
        vst1q_u16(s+r*16,a); vst1q_u16(s+r*16+8,b);
    }
}
static const int DIRS8[8][3]={{1,0,40},{-1,0,40},{0,1,20},{0,-1,20},
                              {1,1,10},{-1,-1,10},{1,-1,10},{-1,1,10}};
static void aggregate(const uint8_t *C,uint16_t *S,int W,int H){
    for (int p=0;p<8;p++){
        int dx=DIRS8[p][0],dy=DIRS8[p][1];
        uint8x16_t p1v=vdupq_n_u8((uint8_t)DIRS8[p][2]);
        if (dy==0){
            #pragma omp parallel for schedule(static)
            for (int y=0;y<H;y++){
                uint8x16_t L[DREG];
                int x0=dx<0?W-1:0,x1=dx<0?-1:W,xs=dx<0?-1:1;
                for (int x=x0;x!=x1;x+=xs){
                    const uint8_t *c=C+((size_t)y*W+x)*D;
                    if (x==x0){ for(int r=0;r<DREG;r++) L[r]=vld1q_u8(c+r*16); }
                    else { uint8x16_t nx[DREG]; step_neon(L,c,nx,p1v);
                           for(int r=0;r<DREG;r++) L[r]=nx[r]; }
                    s_add(S+((size_t)y*W+x)*D,L);
                }
            }
        } else {
            uint8_t *La=malloc((size_t)W*D), *Lb=malloc((size_t)W*D);
            memset(La,0,(size_t)W*D);
            int y0=dy<0?H-1:0,y1=dy<0?-1:H,ys=dy<0?-1:1;
            for (int y=y0;y!=y1;y+=ys){
                #pragma omp parallel for schedule(static)
                for (int x=0;x<W;x++){
                    const uint8_t *c=C+((size_t)y*W+x)*D;
                    uint8_t *cur=Lb+(size_t)x*D;
                    int px=x-dx,py=y-dy;
                    if (px>=0&&px<W&&py>=0&&py<H){
                        uint8x16_t pv[DREG],nx[DREG];
                        const uint8_t *pp=La+(size_t)px*D;
                        for(int r=0;r<DREG;r++) pv[r]=vld1q_u8(pp+r*16);
                        step_neon(pv,c,nx,p1v);
                        for(int r=0;r<DREG;r++) vst1q_u8(cur+r*16,nx[r]);
                    } else memcpy(cur,c,D);
                    { uint8x16_t Lr[DREG];
                      for(int r=0;r<DREG;r++) Lr[r]=vld1q_u8(cur+r*16);
                      s_add(S+((size_t)y*W+x)*D,Lr); }
                }
                uint8_t *t=La; La=Lb; Lb=t;
            }
            free(La); free(Lb);
        }
    }
}
static uint8_t *stage_cost(const uint8_t *L,const uint8_t *R,int W,int H){
    uint8_t *sl=malloc((size_t)W*H),*sr=malloc((size_t)W*H);
    uint64_t *cl=malloc((size_t)W*H*8),*cr=malloc((size_t)W*H*8);
    uint8_t *C=malloc((size_t)W*H*D);
    sobelx(L,sl,W,H); sobelx(R,sr,W,H);
    census_neon(sl,cl,W,H); census_neon(sr,cr,W,H);
    costvol(cl,cr,C,W,H);
    free(sl);free(sr);free(cl);free(cr);
    return C;
}
int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: %s L R [-g golden] [-o out] [-n N] [-t threads]\n",argv[0]);return 1;}
    const char *gp=NULL,*op=NULL; int reps=5;
    for(int i=3;i<argc;i++){
        if(!strcmp(argv[i],"-g")&&i+1<argc)gp=argv[++i];
        else if(!strcmp(argv[i],"-o")&&i+1<argc)op=argv[++i];
        else if(!strcmp(argv[i],"-n")&&i+1<argc)reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-t")&&i+1<argc){int t=atoi(argv[++i]);
#ifdef _OPENMP
            if(t>0) omp_set_num_threads(t);
#endif
        }
    }
    int W,H,W2,H2;
    uint8_t *L=pgm_read(argv[1],&W,&H),*R=pgm_read(argv[2],&W2,&H2);
    if(!L||!R||W!=W2||H!=H2){fprintf(stderr,"bad input\n");return 1;}
    int hw=W/2,hh=H/2,ow=W/2,oh=H/2;
    uint8_t *disp=malloc((size_t)ow*oh);
    double *ms=malloc(sizeof(double)*reps);
    int th=1;
#ifdef _OPENMP
    #pragma omp parallel
    { if(omp_get_thread_num()==0) th=omp_get_num_threads(); }
#endif
    for(int r=0;r<reps;r++){
        double t0=now_ms();
        uint8_t *Lh=malloc((size_t)hw*hh),*Rh=malloc((size_t)hw*hh);
        half(L,Lh,W,H); half(R,Rh,W,H);
        uint8_t *C1=stage_cost(Lh,Rh,hw,hh);
        uint16_t *S1=calloc((size_t)hw*hh*D,2);
        aggregate(C1,S1,hw,hh);
        #pragma omp parallel for schedule(static)
        for (long i=0;i<(long)((size_t)hw*hh);i++){
            const uint16_t *s=S1+(size_t)i*D; uint16_t bv=s[0]; int bd=0;
            for(int d=1;d<D;d++) if(s[d]<bv){bv=s[d];bd=d;}
            (void)bd;
        }
        free(S1);
        uint8_t *C2=stage_cost(L,R,W,H);
        costavg(C2,C1,W,H);
        free(C1);
        uint16_t *S2=calloc((size_t)W*H*D,2);
        aggregate(C2,S2,W,H);
        free(C2);
        #pragma omp parallel for schedule(static)
        for (int y=0;y<oh;y++) for (int x=0;x<ow;x++){
            const uint16_t *s=S2+((size_t)(2*y)*W+(2*x))*D;
            uint16_t bv=s[0]; int bd=0;
            for(int d=1;d<D;d++) if(s[d]<bv){bv=s[d];bd=d;}
            disp[(size_t)y*ow+x]=(uint8_t)bd;
        }
        free(S2); free(Lh); free(Rh);
        ms[r]=now_ms()-t0;
    }
    for(int i=0;i<reps;i++)for(int j=i+1;j<reps;j++)
        if(ms[j]<ms[i]){double t=ms[i];ms[i]=ms[j];ms[j]=t;}
    double med=ms[reps/2];
    uint64_t h=fnv1a64(disp,(size_t)ow*oh);
    int gm=-1;
    if(gp){int gw,gh;uint8_t*G=pgm_read(gp,&gw,&gh);
        if(G&&gw==ow&&gh==oh)gm=!memcmp(G,disp,(size_t)ow*oh);else gm=0;}
    double workpx=(double)hw*hh+(double)W*H;
    printf("ms_neon  in %dx%d out %dx%d D=%d paths=8 th=%d  median %.2f ms  fps %.2f  "
           "MDE/s(work) %.0f  hash %016llx%s\n",
           W,H,ow,oh,D,th,med,1000.0/med,workpx*D/med/1000.0,
           (unsigned long long)h,gm==1?"  GOLDEN OK":gm==0?"  GOLDEN FAIL":"");
    if(op) pgm_write(op,disp,ow,oh);
    return gm==0?2:0;
}
