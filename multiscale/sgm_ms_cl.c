/* multiscale/sgm_ms_cl.c — Configuration B host for Mali. Bit-exact against
 * multiscale/sgm_ms_ref.c's golden; same protocol as the other B targets. */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/sgm.h"
#define D 64
#define CK(v,w) do{ if((v)!=CL_SUCCESS){fprintf(stderr,"OpenCL %s: %d\n",w,(int)(v));return 1;} }while(0)

static char *slurp(const char *p,size_t *n){FILE*f=fopen(p,"rb");if(!f)return 0;
    fseek(f,0,SEEK_END);long s=ftell(f);fseek(f,0,SEEK_SET);
    char*b=malloc(s+1);*n=fread(b,1,s,f);b[*n]=0;fclose(f);return b;}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: %s L R [-g golden] [-o out] [-n N]\n",argv[0]);return 1;}
    const char *gp=NULL,*op=NULL; int reps=6;
    for(int i=3;i<argc;i++){
        if(!strcmp(argv[i],"-g")&&i+1<argc)gp=argv[++i];
        else if(!strcmp(argv[i],"-o")&&i+1<argc)op=argv[++i];
        else if(!strcmp(argv[i],"-n")&&i+1<argc)reps=atoi(argv[++i]);
    }
    int W,H,W2,H2;
    uint8_t *L=pgm_read(argv[1],&W,&H),*R=pgm_read(argv[2],&W2,&H2);
    if(!L||!R||W!=W2||H!=H2){fprintf(stderr,"bad input\n");return 1;}
    int hw=W/2,hh=H/2,ow=W/2,oh=H/2;
    size_t N=(size_t)W*H,Nh=(size_t)hw*hh;

    cl_int e; cl_platform_id pf; cl_device_id dv;
    CK(clGetPlatformIDs(1,&pf,0),"platform");
    CK(clGetDeviceIDs(pf,CL_DEVICE_TYPE_GPU,1,&dv,0),"device");
    char nm[128]={0}; clGetDeviceInfo(dv,CL_DEVICE_NAME,sizeof nm,nm,0);
    fprintf(stderr,"# opencl device: %s\n",nm);
    cl_context cx=clCreateContext(0,1,&dv,0,0,&e); CK(e,"ctx");
    cl_command_queue q=clCreateCommandQueueWithProperties(cx,dv,0,&e); CK(e,"q");
    size_t sn; const char *src_path=getenv("SGM_MS_CL")?getenv("SGM_MS_CL"):"multiscale/sgm_ms.cl";
    char *src=slurp(src_path,&sn); if(!src){fprintf(stderr,"no %s\n",src_path);return 1;}
    cl_program pr=clCreateProgramWithSource(cx,1,(const char**)&src,&sn,&e); CK(e,"prog");
    if(clBuildProgram(pr,1,&dv,"",0,0)!=CL_SUCCESS){
        size_t ln; clGetProgramBuildInfo(pr,dv,CL_PROGRAM_BUILD_LOG,0,0,&ln);
        char*lg=malloc(ln+1); clGetProgramBuildInfo(pr,dv,CL_PROGRAM_BUILD_LOG,ln,lg,0);
        lg[ln]=0; fprintf(stderr,"build:\n%s\n",lg); return 1; }
    const char *kn[]={"sobelx","halfscale","census","cost_build","cost_avg","agg_line","argmin_half"};
    cl_kernel K[7];
    for(int i=0;i<7;i++){K[i]=clCreateKernel(pr,kn[i],&e);CK(e,kn[i]);}
    #define KSOB K[0]
    #define KHALF K[1]
    #define KCEN K[2]
    #define KCOST K[3]
    #define KAVG K[4]
    #define KAGG K[5]
    #define KARG K[6]

    cl_mem bL=clCreateBuffer(cx,CL_MEM_READ_ONLY,N,0,&e);
    cl_mem bR=clCreateBuffer(cx,CL_MEM_READ_ONLY,N,0,&e);
    cl_mem bLh=clCreateBuffer(cx,CL_MEM_READ_WRITE,Nh,0,&e);
    cl_mem bRh=clCreateBuffer(cx,CL_MEM_READ_WRITE,Nh,0,&e);
    cl_mem bS1=clCreateBuffer(cx,CL_MEM_READ_WRITE,N,0,&e);
    cl_mem bS2=clCreateBuffer(cx,CL_MEM_READ_WRITE,N,0,&e);
    cl_mem bcl=clCreateBuffer(cx,CL_MEM_READ_WRITE,N*8,0,&e);
    cl_mem bcr=clCreateBuffer(cx,CL_MEM_READ_WRITE,N*8,0,&e);
    cl_mem bC1=clCreateBuffer(cx,CL_MEM_READ_WRITE,Nh*D,0,&e);
    cl_mem bC2=clCreateBuffer(cx,CL_MEM_READ_WRITE,N*D,0,&e);
    cl_mem bS=clCreateBuffer(cx,CL_MEM_READ_WRITE,N*D*2,0,&e);
    cl_mem bD=clCreateBuffer(cx,CL_MEM_WRITE_ONLY,(size_t)ow*oh,0,&e); CK(e,"bufs");
    CK(clEnqueueWriteBuffer(q,bL,CL_TRUE,0,N,L,0,0,0),"wrL");
    CK(clEnqueueWriteBuffer(q,bR,CL_TRUE,0,N,R,0,0,0),"wrR");

    uint8_t *disp=malloc((size_t)ow*oh);
    double *ms=malloc(sizeof(double)*reps);
    static const int DIRS8[8][3]={{1,0,40},{-1,0,40},{0,1,20},{0,-1,20},
                                  {1,1,10},{-1,-1,10},{1,-1,10},{-1,1,10}};
    for(int r=0;r<reps+2;r++){
        double t0=now_ms();
        /* helpers */
        #define RUN2(k,W_,H_) do{ size_t g[2]={((size_t)(W_)+15)/16*16,((size_t)(H_)+3)/4*4}, l[2]={16,4}; \
            CK(clEnqueueNDRangeKernel(q,k,2,0,g,l,0,0,0),"k2d"); }while(0)
        #define SA(k,i,v) clSetKernelArg(k,i,sizeof(v),&(v))
        int Wf=W,Hf=H,Whh=hw,Hhh=hh;
        /* downsample from the RAW image (as the oracle does) */
        SA(KHALF,0,bL);SA(KHALF,1,bLh);SA(KHALF,2,Wf);SA(KHALF,3,Hf); RUN2(KHALF,W,H);
        SA(KHALF,0,bR);SA(KHALF,1,bRh);SA(KHALF,2,Wf);SA(KHALF,3,Hf); RUN2(KHALF,W,H);
        /* stage 1 costs at half res */
        SA(KSOB,0,bLh);SA(KSOB,1,bS1);SA(KSOB,2,Whh);SA(KSOB,3,Hhh); RUN2(KSOB,hw,hh);
        SA(KCEN,0,bS1);SA(KCEN,1,bcl);SA(KCEN,2,Whh);SA(KCEN,3,Hhh); RUN2(KCEN,hw,hh);
        SA(KSOB,0,bRh);SA(KSOB,1,bS2);SA(KSOB,2,Whh);SA(KSOB,3,Hhh); RUN2(KSOB,hw,hh);
        SA(KCEN,0,bS2);SA(KCEN,1,bcr);SA(KCEN,2,Whh);SA(KCEN,3,Hhh); RUN2(KCEN,hw,hh);
        SA(KCOST,0,bcl);SA(KCOST,1,bcr);SA(KCOST,2,bC1);SA(KCOST,3,Whh);SA(KCOST,4,Hhh); RUN2(KCOST,hw,hh);
        /* stage 1 aggregation + argmin (work performed; result unused) */
        { cl_uchar z=0; CK(clEnqueueFillBuffer(q,bS,&z,1,0,Nh*D*2,0,0,0),"fill1"); }
        for(int p=0;p<8;p++){
            int sx=DIRS8[p][0],sy=DIRS8[p][1],p1=DIRS8[p][2];
            int nl = sy==0?Hhh:(sx==0?Whh:Whh+Hhh-1);
            SA(KAGG,0,bC1);SA(KAGG,1,bS);SA(KAGG,2,Whh);SA(KAGG,3,Hhh);
            SA(KAGG,4,sx);SA(KAGG,5,sy);SA(KAGG,6,p1);SA(KAGG,7,nl);
            size_t g=(size_t)nl*16, l=16;
            CK(clEnqueueNDRangeKernel(q,KAGG,1,0,&g,&l,0,0,0),"agg1");
        }
        SA(KARG,0,bS);SA(KARG,1,bD);SA(KARG,2,Whh);SA(KARG,3,Hhh); RUN2(KARG,hw/2,hh/2);
        /* stage 2 costs at full res, averaged with stage 1 */
        SA(KSOB,0,bL);SA(KSOB,1,bS1);SA(KSOB,2,Wf);SA(KSOB,3,Hf); RUN2(KSOB,W,H);
        SA(KCEN,0,bS1);SA(KCEN,1,bcl);SA(KCEN,2,Wf);SA(KCEN,3,Hf); RUN2(KCEN,W,H);
        SA(KSOB,0,bR);SA(KSOB,1,bS2);SA(KSOB,2,Wf);SA(KSOB,3,Hf); RUN2(KSOB,W,H);
        SA(KCEN,0,bS2);SA(KCEN,1,bcr);SA(KCEN,2,Wf);SA(KCEN,3,Hf); RUN2(KCEN,W,H);
        SA(KCOST,0,bcl);SA(KCOST,1,bcr);SA(KCOST,2,bC2);SA(KCOST,3,Wf);SA(KCOST,4,Hf); RUN2(KCOST,W,H);
        SA(KAVG,0,bC2);SA(KAVG,1,bC1);SA(KAVG,2,Wf);SA(KAVG,3,Hf); RUN2(KAVG,W,H);
        { cl_uchar z=0; CK(clEnqueueFillBuffer(q,bS,&z,1,0,N*D*2,0,0,0),"fill2"); }
        for(int p=0;p<8;p++){
            int sx=DIRS8[p][0],sy=DIRS8[p][1],p1=DIRS8[p][2];
            int nl = sy==0?Hf:(sx==0?Wf:Wf+Hf-1);
            SA(KAGG,0,bC2);SA(KAGG,1,bS);SA(KAGG,2,Wf);SA(KAGG,3,Hf);
            SA(KAGG,4,sx);SA(KAGG,5,sy);SA(KAGG,6,p1);SA(KAGG,7,nl);
            size_t g=(size_t)nl*16, l=16;
            CK(clEnqueueNDRangeKernel(q,KAGG,1,0,&g,&l,0,0,0),"agg2");
        }
        SA(KARG,0,bS);SA(KARG,1,bD);SA(KARG,2,Wf);SA(KARG,3,Hf); RUN2(KARG,ow,oh);
        CK(clEnqueueReadBuffer(q,bD,CL_TRUE,0,(size_t)ow*oh,disp,0,0,0),"rd");
        if(r>=2) ms[r-2]=now_ms()-t0;
    }
    for(int i=0;i<reps;i++)for(int j=i+1;j<reps;j++)
        if(ms[j]<ms[i]){double t=ms[i];ms[i]=ms[j];ms[j]=t;}
    double med=ms[reps/2];
    uint64_t h=fnv1a64(disp,(size_t)ow*oh);
    int gm=-1;
    if(gp){int gw,gh;uint8_t*G=pgm_read(gp,&gw,&gh);
        if(G&&gw==ow&&gh==oh)gm=!memcmp(G,disp,(size_t)ow*oh);else gm=0;}
    double workpx=(double)hw*hh+(double)W*H;
    printf("ms_cl  in %dx%d out %dx%d D=%d paths=8  median %.2f ms  fps %.2f  "
           "MDE/s(work) %.0f  hash %016llx%s\n",
           W,H,ow,oh,D,med,1000.0/med,workpx*D/med/1000.0,
           (unsigned long long)h, gm==1?"  GOLDEN OK":gm==0?"  GOLDEN FAIL":"");
    if(op) pgm_write(op,disp,ow,oh);
    return gm==0?2:0;
}
