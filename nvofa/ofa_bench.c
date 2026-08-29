/* ofa_bench.c — time the RTX 5090's fixed-function stereo engine (NVOFA).
 *
 * Same measurement shape as the Jetson OFA numbers: warm-up, then the median of
 * N timed iterations of submit+sync only. NOT bit-exact to our golden -- this is
 * NVIDIA's own stereo implementation with its own cost function -- so it is a
 * throughput comparison at matched resolution and disparity range, nothing more.
 *
 * Uses NV_OF_MODE_STEREODISPARITY, which NVIDIA documents as deprecated in
 * favour of reading the X component of general optical flow. Measured now
 * because the mode still exists and is the like-for-like comparison to the
 * Jetson OFA rows; recorded as using a mode that is being retired.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <dlfcn.h>
#include <cuda.h>
/* NV_OF_ROI_RECT is referenced by NV_OF_EXECUTE_INPUT_PARAMS before its own
 * definition in this header revision; forward-declare it. */
typedef struct _NV_OF_ROI_RECT NV_OF_ROI_RECT;
#include "nvOpticalFlowCommon.h"
#include "nvOpticalFlowCuda.h"

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e3 + t.tv_nsec/1e6; }
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y); }

static unsigned char *pgm(const char *p,int *W,int *H){
    FILE*f=fopen(p,"rb"); if(!f) return NULL; char m[3]; int mx;
    if(fscanf(f,"%2s %d %d %d",m,W,H,&mx)!=4){fclose(f);return NULL;}
    fgetc(f); unsigned char*d=malloc((size_t)*W**H);
    if(fread(d,1,(size_t)*W**H,f)!=(size_t)*W**H){free(d);fclose(f);return NULL;}
    fclose(f); return d; }

#define CK(x,s) do{ NV_OF_STATUS r=(x); if(r!=NV_OF_SUCCESS){fprintf(stderr,"%s failed: %d\n",s,r); return 1;} }while(0)

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: %s L.pgm R.pgm [range 128|256] [n]\n",argv[0]);return 1;}
    int range = argc>3?atoi(argv[3]):128, N = argc>4?atoi(argv[4]):60;
    int W,H,W2,H2; unsigned char*L=pgm(argv[1],&W,&H),*R=pgm(argv[2],&W2,&H2);
    if(!L||!R||W!=W2||H!=H2){fprintf(stderr,"bad input\n");return 1;}

    cuInit(0); CUdevice dev; CUcontext ctx;
    cuDeviceGet(&dev,0); cuCtxCreate(&ctx,0,dev);
    char nm[128]; cuDeviceGetName(nm,sizeof nm,dev);

    void*so=dlopen("libnvidia-opticalflow.so.1",RTLD_LAZY);
    if(!so){fprintf(stderr,"dlopen: %s\n",dlerror());return 1;}
    NV_OF_STATUS (*create)(uint32_t,NV_OF_CUDA_API_FUNCTION_LIST*) =
        (NV_OF_STATUS(*)(uint32_t,NV_OF_CUDA_API_FUNCTION_LIST*))dlsym(so,"NvOFAPICreateInstanceCuda");
    if(!create){fprintf(stderr,"no NvOFAPICreateInstanceCuda\n");return 1;}
    NV_OF_CUDA_API_FUNCTION_LIST api; memset(&api,0,sizeof api);
    CK(create(NV_OF_API_VERSION,&api),"CreateInstanceCuda");

    NvOFHandle h; CK(api.nvCreateOpticalFlowCuda(ctx,&h),"nvCreateOpticalFlowCuda");

    /* Ask the device what it actually supports before assuming. */
    { uint32_t n=0, v[16]; NV_OF_STATUS st;
      st=api.nvOFGetCaps(h,NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES,NULL,&n);
      if(st==NV_OF_SUCCESS && n && n<=16){ api.nvOFGetCaps(h,NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES,v,&n);
        printf("  supported output grid sizes:"); for(uint32_t i=0;i<n;i++) printf(" %u",v[i]); printf("\n"); }
      n=1; uint32_t one=0;
      if(api.nvOFGetCaps(h,NV_OF_CAPS_SUPPORT_HINT_WITH_ST_MODE,&one,&n)==NV_OF_SUCCESS)
        printf("  stereo-mode hint cap reported: %u\n", one);
    }

    NV_OF_INIT_PARAMS ip; memset(&ip,0,sizeof ip);
    ip.width=W; ip.height=H;
    int gs = getenv("OF_GRID")?atoi(getenv("OF_GRID")):1;
    ip.outGridSize=(gs==4)?NV_OF_OUTPUT_VECTOR_GRID_SIZE_4:(gs==2)?NV_OF_OUTPUT_VECTOR_GRID_SIZE_2:NV_OF_OUTPUT_VECTOR_GRID_SIZE_1;
    ip.hintGridSize=NV_OF_HINT_VECTOR_GRID_SIZE_1;
    const char *modestr = getenv("OF_MODE");
    int use_of = (modestr && !strcmp(modestr,"of"));
    ip.mode = use_of ? NV_OF_MODE_OPTICALFLOW : NV_OF_MODE_STEREODISPARITY;
    ip.perfLevel=NV_OF_PERF_LEVEL_SLOW;   /* highest quality, like our 4-path config */
    ip.disparityRange = (range==256)?NV_OF_STEREO_DISPARITY_RANGE_256:NV_OF_STEREO_DISPARITY_RANGE_128;
    CK(api.nvOFInit(h,&ip),"nvOFInit");

    NV_OF_BUFFER_DESCRIPTOR din={.width=W,.height=H,.bufferUsage=NV_OF_BUFFER_USAGE_INPUT,
                                 .bufferFormat=NV_OF_BUFFER_FORMAT_GRAYSCALE8};
    NV_OF_BUFFER_DESCRIPTOR dout={.width=W,.height=H,.bufferUsage=NV_OF_BUFFER_USAGE_OUTPUT,
                                  .bufferFormat=NV_OF_BUFFER_FORMAT_SHORT};
    if(use_of) dout.bufferFormat=NV_OF_BUFFER_FORMAT_SHORT2;
    NvOFGPUBufferHandle bL,bR,bO;
    CK(api.nvOFCreateGPUBufferCuda(h,&din,NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,&bL),"buf L");
    CK(api.nvOFCreateGPUBufferCuda(h,&din,NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,&bR),"buf R");
    CK(api.nvOFCreateGPUBufferCuda(h,&dout,NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,&bO),"buf O");

    CUdeviceptr pL,pR; NV_OF_CUDA_BUFFER_STRIDE_INFO si;
    pL = api.nvOFGPUBufferGetCUdeviceptr(bL); pR = api.nvOFGPUBufferGetCUdeviceptr(bR);
    api.nvOFGPUBufferGetStrideInfo(bL,&si);
    size_t pitch = si.strideInfo[0].strideXInBytes;
    for(int y=0;y<H;y++){ cuMemcpyHtoD(pL+y*pitch,L+(size_t)y*W,W);
                          cuMemcpyHtoD(pR+y*pitch,R+(size_t)y*W,W); }

    NV_OF_EXECUTE_INPUT_PARAMS  ein; memset(&ein,0,sizeof ein);
    NV_OF_EXECUTE_OUTPUT_PARAMS eout; memset(&eout,0,sizeof eout);
    ein.inputFrame=bL; ein.referenceFrame=bR; ein.disableTemporalHints=NV_OF_TRUE;
    eout.outputBuffer=bO;

    for(int i=0;i<10;i++) CK(api.nvOFExecute(h,&ein,&eout),"warm execute");
    cuCtxSynchronize();
    double *ms=malloc(sizeof(double)*N);
    for(int i=0;i<N;i++){ double t0=now_ms();
        CK(api.nvOFExecute(h,&ein,&eout),"execute"); cuCtxSynchronize();
        ms[i]=now_ms()-t0; }
    /* Read back the flow field and test the deprecation's premise directly:
     * does the X component of general optical flow equal stereo disparity?
     * Output is SHORT2 in Q5 (1/32 pel). Any non-zero Y on a rectified pair is
     * by definition an error -- stereo mode could not produce one. */
    { CUdeviceptr pO = api.nvOFGPUBufferGetCUdeviceptr(bO);
      NV_OF_CUDA_BUFFER_STRIDE_INFO so; api.nvOFGPUBufferGetStrideInfo(bO,&so);
      size_t op = so.strideInfo[0].strideXInBytes;
      short *row = malloc(op);
      double sx=0, sy=0, aymax=0; long n=0, ynz=0;
      FILE *f = fopen("/tmp/ofa_dx.raw","wb");
      for(int y=0;y<H;y++){
        cuMemcpyDtoH(row, pO + (size_t)y*op, op);
        for(int x=0;x<W;x++){
          double dx = row[2*x]/32.0, dy = row[2*x+1]/32.0;
          sx += dx; sy += dy; if(dy!=0){ ynz++; if(fabs(dy)>aymax) aymax=fabs(dy); }
          unsigned char v = (unsigned char)(dx < 0 ? -dx : dx); fwrite(&v,1,1,f); n++;
        }
      }
      fclose(f); free(row);
      printf("  flow field: mean dx %.2f  mean dy %.3f  |dy|max %.2f  non-zero dy %.1f%% of pixels\n",
             sx/n, sy/n, aymax, 100.0*ynz/n);
    }
    qsort(ms,N,sizeof(double),cmpd);
    double med=ms[N/2];
    printf("NVOFA  %s  %dx%d  disparityRange=%d  median %.3f ms  min %.3f  p95 %.3f  fps %.1f  MDE/s %.0f\n",
           nm,W,H,range,med,ms[0],ms[(int)(N*0.95)],1000.0/med,(double)W*H*range/med/1000.0);
    return 0;
}
