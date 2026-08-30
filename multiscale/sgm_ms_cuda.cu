/* multiscale/sgm_ms_cuda.cu — Configuration B, CUDA.
 *
 * Same shape as cuda/sgm_cuda_opt.cu (a warp owns a line's disparity range, L
 * in registers, shuffle reductions, zero barriers) extended with what
 * Configuration B needs: the SobelX prefilter, TWO full scales with averaged
 * costs, EIGHT paths with direction-weighted P1 (40/20/10), P2=200 saturating,
 * and a half-resolution output map. Diagonal paths use a generic line kernel:
 * a warp per diagonal line, marching (x,y) together.
 *
 * Correctness gate: bit-exact against multiscale/sgm_ms_ref.c's golden.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern "C" {
#include "../common/sgm.h"
}
#define D 64
#define WARP 32
#define DPT (D/WARP)
#define P2V 200
#define CINVAL 63
#define FULL 0xffffffffu

#define CK(v) do{cudaError_t e_=(v); if(e_!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_));exit(1);} }while(0)

__device__ __forceinline__ int clampi(int v,int lo,int hi){return v<lo?lo:(v>hi?hi:v);}

__global__ void k_sobel(const unsigned char*__restrict__ im, unsigned char*__restrict__ out,int W,int H){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=W||y>=H) return;
    int xm=clampi(x-1,0,W-1),xp=clampi(x+1,0,W-1),ym=clampi(y-1,0,H-1),yp=clampi(y+1,0,H-1);
    int gx=-im[ym*W+xm]+im[ym*W+xp]-2*im[y*W+xm]+2*im[y*W+xp]-im[yp*W+xm]+im[yp*W+xp];
    out[y*W+x]=(unsigned char)clampi(128+(gx>>3),0,255);
}
__global__ void k_half(const unsigned char*__restrict__ im, unsigned char*__restrict__ out,int W,int H){
    int w=W/2,h=H/2, x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=w||y>=h) return;
    out[y*w+x]=(unsigned char)((im[(2*y)*W+2*x]+im[(2*y)*W+2*x+1]
                               +im[(2*y+1)*W+2*x]+im[(2*y+1)*W+2*x+1]+2)>>2);
}
__global__ void k_census(const unsigned char*__restrict__ im, unsigned long long*__restrict__ out,int W,int H){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=W||y>=H) return;
    unsigned char c=im[y*W+x]; unsigned long long s=0; int n=0;
    for (int dy=-3;dy<=3;dy++){ int yy=clampi(y+dy,0,H-1);
        for (int dx=-4;dx<=4;dx++){ if(!dx&&!dy) continue;
            if (im[yy*W+clampi(x+dx,0,W-1)]<c) s|=1ULL<<n;
            n++; } }
    out[(size_t)y*W+x]=s;
}
__global__ void k_cost(const unsigned long long*__restrict__ cl,const unsigned long long*__restrict__ cr,
                       unsigned char*__restrict__ C,int W,int H){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=W||y>=H) return;
    unsigned long long a=cl[(size_t)y*W+x];
    unsigned char *c=C+((size_t)y*W+x)*D;
    for (int d=0;d<D;d++){ int xr=x-d;
        c[d]=(xr<0)?(unsigned char)CINVAL:(unsigned char)__popcll(a^cr[(size_t)y*W+xr]); }
}
__global__ void k_avg(unsigned char*__restrict__ C2,const unsigned char*__restrict__ C1,int W,int H){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=W||y>=H) return;
    unsigned char *c2=C2+((size_t)y*W+x)*D;
    const unsigned char *c1=C1+((size_t)(y/2)*(W/2)+(x/2))*D;
    for (int d=0;d<D;d++) c2[d]=(unsigned char)((c2[d]+c1[d/2]+1)>>1);
}
__device__ __forceinline__ unsigned int warp_min(unsigned int v){
    for (int o=WARP/2;o>0;o>>=1) v=min(v,__shfl_xor_sync(FULL,v,o));
    return v;
}
__device__ __forceinline__ void cost_lane(const unsigned char*__restrict__ C,size_t px,int d0,unsigned char *c){
    const unsigned char *s=C+px*D+d0;
    for (int k=0;k<DPT;k++) c[k]=s[k];
}
__device__ __forceinline__ void step_warp(const unsigned char *prev,const unsigned char *c,
                                          unsigned char *cur,unsigned int mn,int lane,int p1){
    unsigned int loNb=__shfl_up_sync(FULL,(unsigned int)prev[DPT-1],1);
    unsigned int hiNb=__shfl_down_sync(FULL,(unsigned int)prev[0],1);
    if (lane==0) loNb=255; if (lane==WARP-1) hiNb=255;
    unsigned int e=min(mn+P2V,255u);
    for (int k=0;k<DPT;k++){
        unsigned int lo=(k==0)?loNb:(unsigned int)prev[k-1];
        unsigned int hi=(k==DPT-1)?hiNb:(unsigned int)prev[k+1];
        unsigned int m=prev[k];
        m=min(m,min(lo+p1,255u)); m=min(m,min(hi+p1,255u)); m=min(m,e);
        cur[k]=(unsigned char)min((unsigned int)c[k]+(m-mn),255u);
    }
}
/* Generic path kernel: one WARP per line. A line is defined by the march
 * direction (sx,sy); its start follows from the line index:
 *   horizontal (sy==0): H lines, start (x0, line)
 *   vertical   (sx==0): W lines, start (line, y0)
 *   diagonal:  W+H-1 lines -- starts down the entry column (x0, line) for the
 *              first H, then along the entry row (x0+sx*(line-H+1), y0)
 * where x0/y0 are the entry edge for the sign of sx/sy. L lives in registers;
 * the min-over-D is a warp shuffle; no __syncthreads anywhere. */
__global__ void k_path(const unsigned char*__restrict__ C, unsigned short*__restrict__ S,
                       int W,int H,int sx,int sy,int p1,int nlines){
    int lane=threadIdx.x, line=blockIdx.x*blockDim.y+threadIdx.y;
    if (line>=nlines) return;
    int x0 = sx>0 ? 0 : (sx<0 ? W-1 : 0);
    int y0 = sy>0 ? 0 : (sy<0 ? H-1 : 0);
    int x,y;
    if (sy==0)      { x=x0; y=line; }
    else if (sx==0) { x=line; y=y0; }
    else if (line<H){ x=x0; y=line; }
    else            { x=x0+sx*(line-H+1); y=y0; }
    int d0=lane*DPT;
    unsigned char L[DPT],c[DPT],nx[DPT];
    int first=1;
    while (x>=0&&x<W&&y>=0&&y<H){
        size_t px=(size_t)y*W+x;
        cost_lane(C,px,d0,c);
        if (first){ for(int k=0;k<DPT;k++) L[k]=c[k]; first=0; }
        else {
            unsigned int mn=255;
            for (int k=0;k<DPT;k++) mn=min(mn,(unsigned int)L[k]);
            mn=warp_min(mn);
            step_warp(L,c,nx,mn,lane,p1);
            for (int k=0;k<DPT;k++) L[k]=nx[k];
        }
        unsigned short *s=S+px*D+d0;
        for (int k=0;k<DPT;k++) s[k]+=L[k];
        x+=sx; y+=sy;
    }
}
__global__ void k_argmin_half(const unsigned short*__restrict__ S, unsigned char*__restrict__ disp,int W,int H){
    int ow=W/2,oh=H/2, x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=ow||y>=oh) return;
    const unsigned short *s=S+((size_t)(2*y)*W+(2*x))*D;
    unsigned short bv=s[0]; int bd=0;
    for (int d=1;d<D;d++) if (s[d]<bv){bv=s[d];bd=d;}
    disp[(size_t)y*ow+x]=(unsigned char)bd;
}

static void full_sgm_costs(const unsigned char *dL,const unsigned char *dR,
                           unsigned char *dSob1,unsigned char *dSob2,
                           unsigned long long *dcl,unsigned long long *dcr,
                           unsigned char *dC,int W,int H){
    dim3 b(16,8), g((W+15)/16,(H+7)/8);
    k_sobel<<<g,b>>>(dL,dSob1,W,H);
    k_sobel<<<g,b>>>(dR,dSob2,W,H);
    k_census<<<g,b>>>(dSob1,dcl,W,H);
    k_census<<<g,b>>>(dSob2,dcr,W,H);
    k_cost<<<g,b>>>(dcl,dcr,dC,W,H);
}
static const int DIRS8[8][3]={{1,0,40},{-1,0,40},{0,1,20},{0,-1,20},
                              {1,1,10},{-1,-1,10},{1,-1,10},{-1,1,10}};
static void aggregate8(unsigned char *dC,unsigned short *dS,int W,int H){
    const int WPB=4; dim3 wb(WARP,WPB);
    for (int p=0;p<8;p++){
        int sx=DIRS8[p][0],sy=DIRS8[p][1],p1=DIRS8[p][2];
        int nlines = sy==0 ? H : (sx==0 ? W : W+H-1);
        k_path<<<(nlines+WPB-1)/WPB,wb>>>(dC,dS,W,H,sx,sy,p1,nlines);
    }
    CK(cudaDeviceSynchronize());
}
int main(int argc,char**argv){
    if (argc<3){fprintf(stderr,"usage: %s L.pgm R.pgm [-g golden] [-o out] [-n N]\n",argv[0]);return 1;}
    const char *gp=NULL,*op=NULL; int reps=10;
    for (int i=3;i<argc;i++){
        if(!strcmp(argv[i],"-g")&&i+1<argc)gp=argv[++i];
        else if(!strcmp(argv[i],"-o")&&i+1<argc)op=argv[++i];
        else if(!strcmp(argv[i],"-n")&&i+1<argc)reps=atoi(argv[++i]);
    }
    int W,H,W2,H2;
    unsigned char *L=pgm_read(argv[1],&W,&H),*R=pgm_read(argv[2],&W2,&H2);
    if(!L||!R||W!=W2||H!=H2){fprintf(stderr,"bad input\n");return 1;}
    int hw=W/2,hh=H/2, ow=W/2,oh=H/2;
    size_t N=(size_t)W*H, Nh=(size_t)hw*hh;
    cudaDeviceProp pr; cudaGetDeviceProperties(&pr,0);
    fprintf(stderr,"# device: %s (%d SMs)\n",pr.name,pr.multiProcessorCount);
    unsigned char *dL,*dR,*dLh,*dRh,*dS1,*dS2,*dC1,*dC2,*ddisp;
    unsigned long long *dcl,*dcr;
    unsigned short *dSagg;
    CK(cudaMalloc(&dL,N)); CK(cudaMalloc(&dR,N));
    CK(cudaMalloc(&dLh,Nh)); CK(cudaMalloc(&dRh,Nh));
    CK(cudaMalloc(&dS1,N)); CK(cudaMalloc(&dS2,N));
    CK(cudaMalloc(&dcl,N*8)); CK(cudaMalloc(&dcr,N*8));
    CK(cudaMalloc(&dC1,Nh*D)); CK(cudaMalloc(&dC2,N*D));
    CK(cudaMalloc(&dSagg,N*D*sizeof(unsigned short)));
    CK(cudaMalloc(&ddisp,(size_t)ow*oh));
    CK(cudaMemcpy(dL,L,N,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dR,R,N,cudaMemcpyHostToDevice));
    unsigned char *disp=(unsigned char*)malloc((size_t)ow*oh);
    double *ms=(double*)malloc(sizeof(double)*reps);
    for (int r=0;r<reps+3;r++){
        double t0=now_ms();
        dim3 b(16,8), gh((hw+15)/16,(hh+7)/8), gf((W+15)/16,(H+7)/8);
        k_half<<<gf,b>>>(dL,dLh,W,H); k_half<<<gf,b>>>(dR,dRh,W,H);
        /* stage 1: full SGM at half res; argmin included in the defined
         * workload, its result unused. */
        full_sgm_costs(dLh,dRh,dS1,dS2,dcl,dcr,dC1,hw,hh);
        CK(cudaMemset(dSagg,0,Nh*D*sizeof(unsigned short)));
        aggregate8(dC1,dSagg,hw,hh);
        k_argmin_half<<<dim3((hw/2+15)/16,(hh/2+7)/8),b>>>(dSagg,ddisp,hw,hh);
        /* stage 2: full res, costs averaged with stage-1's */
        full_sgm_costs(dL,dR,dS1,dS2,dcl,dcr,dC2,W,H);
        k_avg<<<gf,b>>>(dC2,dC1,W,H);
        CK(cudaMemset(dSagg,0,N*D*sizeof(unsigned short)));
        aggregate8(dC2,dSagg,W,H);
        k_argmin_half<<<dim3((ow+15)/16,(oh+7)/8),b>>>(dSagg,ddisp,W,H);
        CK(cudaMemcpy(disp,ddisp,(size_t)ow*oh,cudaMemcpyDeviceToHost));
        CK(cudaDeviceSynchronize());
        if (r>=3) ms[r-3]=now_ms()-t0;
    }
    for (int i=0;i<reps;i++) for (int j=i+1;j<reps;j++)
        if (ms[j]<ms[i]){double t=ms[i];ms[i]=ms[j];ms[j]=t;}
    double med=ms[reps/2];
    unsigned long long h=fnv1a64(disp,(size_t)ow*oh);
    int gm=-1;
    if (gp){ int gw,gh2; unsigned char *G=pgm_read(gp,&gw,&gh2);
        if (G&&gw==ow&&gh2==oh) gm=!memcmp(G,disp,(size_t)ow*oh); else gm=0; }
    double workpx=(double)hw*hh+(double)W*H;
    printf("ms_cuda  in %dx%d out %dx%d D=%d paths=8  median %.2f ms  fps %.2f  "
           "MDE/s(work) %.0f  hash %016llx%s\n",
           W,H,ow,oh,D,med,1000.0/med,workpx*D/med/1000.0,h,
           gm==1?"  GOLDEN OK":gm==0?"  GOLDEN FAIL":"");
    if (op) pgm_write(op,disp,ow,oh);
    return gm==0?2:0;
}
