/* multiscale/sgm_ms_ref.c — THE ORACLE for Configuration B.
 *
 * A second benchmark configuration, deliberately harder and more
 * embedded-flavoured than the primary one. It exercises features the primary
 * configuration does not:
 *
 *   input 1920x800 grey            output 960x400 disparity (half-res each axis)
 *   TWO SCALES, cost-fused: a full 64-disparity SGM runs at half resolution AND
 *     at full resolution, and the two DATA COSTS ARE AVERAGED before
 *     aggregation (multi-scale cost fusion -- robustness on low-texture
 *     regions; both scales do full work, this is not a coarse-to-fine search).
 *   census 9x7 computed on a HORIZONTAL-SOBEL prefiltered image (gradient-
 *     domain matching, as in OpenCV's SGBM cost).
 *   EIGHT aggregation paths with DIRECTION-WEIGHTED P1: 40 horizontal /
 *     20 vertical / 10 diagonal. P2 = 200, constant.
 *   8-bit path costs, per-pixel renormalisation, SATURATING adds (with 6-bit
 *     costs and P2=200 the bracket can exceed 255, so a rule must exist;
 *     saturation is the pinned choice).
 *   integer disparity out; ties to the lowest d; x-d outside the image costs
 *     63; stage-1 downsample is a 2x2 box mean; the stage merge averages
 *     full-res cost (x,y,d) with half-res cost (x/2,y/2,d/2).
 *   NO left-right check, NO median, NO speckle filter.
 *
 * Same rules as the primary oracle: plain C, no SIMD, no tricks, do not
 * optimise this file. Its output defines "correct" for every Configuration-B
 * implementation.
 *
 * Usage: sgm_ms_ref L.pgm R.pgm [-o out.pgm] [-g golden.pgm] [-n N] [-j out.json]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/sgm.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define D 64
#define P2V 200
#define CBITS 62            /* 9x7, centre skipped, as the main benchmark */
#define CINVAL 63

static inline int clampi(int v,int lo,int hi){return v<lo?lo:(v>hi?hi:v);}
static inline int sat255(int v){return v>255?255:v;}

/* horizontal Sobel prefilter, pinned output mapping */
static void sobelx(const uint8_t *im, uint8_t *out, int W, int H){
    for (int y=0;y<H;y++) for (int x=0;x<W;x++){
        int xm=clampi(x-1,0,W-1), xp=clampi(x+1,0,W-1);
        int ym=clampi(y-1,0,H-1), yp=clampi(y+1,0,H-1);
        int gx = -im[ym*W+xm]+im[ym*W+xp]-2*im[y*W+xm]+2*im[y*W+xp]-im[yp*W+xm]+im[yp*W+xp];
        out[y*W+x]=(uint8_t)clampi(128+(gx>>3),0,255);
    }
}
static void census9x7(const uint8_t *im, uint64_t *out, int W, int H){
    for (int y=0;y<H;y++) for (int x=0;x<W;x++){
        uint8_t c=im[y*W+x]; uint64_t s=0; int n=0;
        for (int dy=-3;dy<=3;dy++){ int yy=clampi(y+dy,0,H-1);
            for (int dx=-4;dx<=4;dx++){ if(!dx&&!dy) continue;
                int xx=clampi(x+dx,0,W-1);
                if (im[yy*W+xx]<c) { s|=(uint64_t)1<<n; }
                n++; } }
        out[y*W+x]=s;
    }
}
static void costvol(const uint64_t *cl,const uint64_t *cr,uint8_t *C,int W,int H){
    for (int y=0;y<H;y++) for (int x=0;x<W;x++){
        uint64_t a=cl[y*W+x]; uint8_t *c=C+((size_t)y*W+x)*D;
        for (int d=0;d<D;d++){ int xr=x-d;
            c[d]=(xr<0)?CINVAL:(uint8_t)__builtin_popcountll(a^cr[y*W+xr]); }
    }
}
/* one recurrence step with a per-direction P1 and saturation */
static void stepP(const uint8_t *prev,const uint8_t *c,uint8_t *cur,int p1){
    int mn=255; for(int d=0;d<D;d++) if(prev[d]<mn) mn=prev[d];
    int e=sat255(mn+P2V);
    for (int d=0;d<D;d++){
        int a=prev[d];
        int b=(d>0)  ?sat255(prev[d-1]+p1):255;
        int f=(d<D-1)?sat255(prev[d+1]+p1):255;
        int m=a; if(b<m)m=b; if(f<m)m=f; if(e<m)m=e;
        cur[d]=(uint8_t)sat255(c[d]+m-mn);
    }
}
static const int DIRS8[8][3]={ /* dx,dy,P1: 40 horizontal, 20 vertical, 10 diagonal */
    { 1,0,40},{-1,0,40},{0, 1,20},{0,-1,20},
    { 1,1,10},{-1,-1,10},{1,-1,10},{-1,1,10}};
/* Parallelism that cannot change a byte of output:
 *  - horizontal paths (dy==0): every ROW is an independent recurrence
 *  - vertical/diagonal paths:  within a row, every COLUMN is independent,
 *    because prev comes from the PREVIOUS row's buffer
 * S accumulation touches each pixel from exactly one line per direction. */
static void aggregate(const uint8_t *C,uint16_t *S,int W,int H){
    for (int p=0;p<8;p++){
        int dx=DIRS8[p][0],dy=DIRS8[p][1],p1=DIRS8[p][2];
        if (dy==0){
            #pragma omp parallel for schedule(static)
            for (int y=0;y<H;y++){
                uint8_t Lrow[2][64]; int cb=0;
                int x0=dx<0?W-1:0,x1=dx<0?-1:W,xs=dx<0?-1:1;
                for (int x=x0;x!=x1;x+=xs){
                    const uint8_t *c=C+((size_t)y*W+x)*D;
                    uint8_t *cur=Lrow[cb];
                    if (x!=x0) stepP(Lrow[cb^1],c,cur,p1);
                    else memcpy(cur,c,D);
                    uint16_t *s=S+((size_t)y*W+x)*D;
                    for (int d=0;d<D;d++) s[d]+=cur[d];
                    cb^=1;
                }
            }
        } else {
            uint8_t *La=malloc((size_t)W*D), *Lb=malloc((size_t)W*D);
            int y0=dy<0?H-1:0,y1=dy<0?-1:H,ys=dy<0?-1:1;
            memset(La,0,(size_t)W*D);
            for (int y=y0;y!=y1;y+=ys){
                #pragma omp parallel for schedule(static)
                for (int x=0;x<W;x++){
                    const uint8_t *c=C+((size_t)y*W+x)*D;
                    uint8_t *cur=Lb+(size_t)x*D;
                    int px=x-dx,py=y-dy;
                    const uint8_t *prev=NULL;
                    if (px>=0&&px<W&&py>=0&&py<H) prev=La+(size_t)px*D;
                    if (prev) stepP(prev,c,cur,p1);
                    else memcpy(cur,c,D);
                    uint16_t *s=S+((size_t)y*W+x)*D;
                    for (int d=0;d<D;d++) s[d]+=cur[d];
                }
                uint8_t *t=La; La=Lb; Lb=t;
            }
            free(La); free(Lb);
        }
    }
}
/* full SGM at one scale: sobel -> census -> cost; returns C (caller frees) */
static uint8_t *stage_cost(const uint8_t *L,const uint8_t *R,int W,int H){
    uint8_t *sl=malloc((size_t)W*H), *sr=malloc((size_t)W*H);
    uint64_t *cl=malloc((size_t)W*H*8), *cr=malloc((size_t)W*H*8);
    uint8_t *C=malloc((size_t)W*H*D);
    sobelx(L,sl,W,H); sobelx(R,sr,W,H);
    census9x7(sl,cl,W,H); census9x7(sr,cr,W,H);
    costvol(cl,cr,C,W,H);
    free(sl);free(sr);free(cl);free(cr);
    return C;
}
static void half(const uint8_t *im,uint8_t *out,int W,int H){ /* 2x2 box mean */
    int w=W/2,h=H/2;
    for (int y=0;y<h;y++) for (int x=0;x<w;x++)
        out[y*w+x]=(uint8_t)((im[(2*y)*W+2*x]+im[(2*y)*W+2*x+1]
                             +im[(2*y+1)*W+2*x]+im[(2*y+1)*W+2*x+1]+2)>>2);
}
int main(int argc,char**argv){
    if (argc<3){fprintf(stderr,"usage: %s L.pgm R.pgm [-o out] [-g golden] [-n N] [-j json]\n",argv[0]);return 1;}
    const char *op=NULL,*gp=NULL,*jp=NULL; int reps=1;
    for (int i=3;i<argc;i++){
        if(!strcmp(argv[i],"-o")&&i+1<argc)op=argv[++i];
        else if(!strcmp(argv[i],"-g")&&i+1<argc)gp=argv[++i];
        else if(!strcmp(argv[i],"-n")&&i+1<argc)reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-j")&&i+1<argc)jp=argv[++i];
    }
    int W,H,W2,H2;
    uint8_t *L=pgm_read(argv[1],&W,&H), *R=pgm_read(argv[2],&W2,&H2);
    if(!L||!R||W!=W2||H!=H2){fprintf(stderr,"bad input\n");return 1;}
    int hw=W/2,hh=H/2, ow=W/2,oh=H/2;
    uint8_t *disp=malloc((size_t)ow*oh);
    double *ms=malloc(sizeof(double)*reps);
    for (int r=0;r<reps;r++){
        double t0=now_ms();
        /* stage 1: full SGM at half res. Its argmin is part of the defined
         * workload (both scales do full work); the result itself is unused. */
        uint8_t *Lh=malloc((size_t)hw*hh),*Rh=malloc((size_t)hw*hh);
        half(L,Lh,W,H); half(R,Rh,W,H);
        uint8_t *C1=stage_cost(Lh,Rh,hw,hh);
        uint16_t *S1=calloc((size_t)hw*hh*D,2);
        aggregate(C1,S1,hw,hh);
        /* stage-1 argmin (work performed; result unused at step=1) */
        for (size_t i=0;i<(size_t)hw*hh;i++){
            const uint16_t *s=S1+i*D; uint16_t bv=s[0]; int bd=0;
            for (int d=1;d<D;d++) if (s[d]<bv){bv=s[d];bd=d;}
            (void)bd;
        }
        free(S1);
        /* stage 2: full res, cost AVERAGED with upsampled stage-1 cost */
        uint8_t *C2=stage_cost(L,R,W,H);
        for (int y=0;y<H;y++) for (int x=0;x<W;x++){
            uint8_t *c2=C2+((size_t)y*W+x)*D;
            const uint8_t *c1=C1+((size_t)(y/2)*hw+(x/2))*D;
            for (int d=0;d<D;d++) c2[d]=(uint8_t)((c2[d]+c1[d/2]+1)>>1);
        }
        free(C1);
        uint16_t *S2=calloc((size_t)W*H*D,2);
        aggregate(C2,S2,W,H);
        free(C2);
        /* argmin at full res, then decimate to the half-res output map */
        for (int y=0;y<oh;y++) for (int x=0;x<ow;x++){
            const uint16_t *s=S2+((size_t)(2*y)*W+(2*x))*D;
            uint16_t bv=s[0]; int bd=0;
            for (int d=1;d<D;d++) if (s[d]<bv){bv=s[d];bd=d;}
            disp[(size_t)y*ow+x]=(uint8_t)bd;
        }
        free(S2); free(Lh); free(Rh);
        ms[r]=now_ms()-t0;
    }
    /* median */
    for (int i=0;i<reps;i++) for (int j=i+1;j<reps;j++)
        if (ms[j]<ms[i]){double t=ms[i];ms[i]=ms[j];ms[j]=t;}
    double med=ms[reps/2];
    uint64_t h=fnv1a64(disp,(size_t)ow*oh);
    int gm=-1;
    if (gp){ int gw,gh; uint8_t *G=pgm_read(gp,&gw,&gh);
        if(G&&gw==ow&&gh==oh){ gm=!memcmp(G,disp,(size_t)ow*oh); free(G);} else gm=0; }
    /* MDE/s under the WORK-PERFORMED convention: both scales count, so
     * (half-res px + full-res px) x D / time */
    double workpx=(double)hw*hh+(double)W*H;
    printf("ms_ref  in %dx%d out %dx%d D=%d paths=8  median %.2f ms  fps %.2f  "
           "MDE/s(work) %.0f  hash %016llx%s\n",
           W,H,ow,oh,D,med,1000.0/med,workpx*D/med/1000.0,
           (unsigned long long)h, gm==1?"  GOLDEN OK":gm==0?"  GOLDEN FAIL":"");
    if (op) pgm_write(op,disp,ow,oh);
    if (jp){ FILE*f=fopen(jp,"a");
        if(f){fprintf(f,"{\"impl\":\"ms_ref\",\"W\":%d,\"H\":%d,\"outW\":%d,\"outH\":%d,"
            "\"D\":%d,\"paths\":8,\"median_ms\":%.3f,\"hash\":\"%016llx\",\"golden_match\":%d}\n",
            W,H,ow,oh,D,med,(unsigned long long)h,gm); fclose(f);} }
    return gm==0?2:0;
}
