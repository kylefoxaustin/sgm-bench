/* sgmms_main.c — ARM host for Configuration B (multiscale SGM) on the NSP.
 * Reads the P5 PGM pair, runs the DSP, hashes the (W/2)x(H/2) output with the
 * same FNV-1a as the sgm-bench harness, and checks it against -G IN THE SAME
 * RUN.  A timing whose hash was not checked in its own run is VOID.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "sgmms.h"
#include "rpcmem.h"
#include "remote.h"
#include "os_defines.h"

static uint64_t fnv1a64(const void *data, size_t n){
    const unsigned char *p = (const unsigned char*)data;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i=0;i<n;i++){ h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1000.0 + t.tv_nsec/1e6; }

static unsigned char *read_pgm(const char *path,int *W,int *H){
    FILE *f=fopen(path,"rb"); if(!f){perror(path);return NULL;}
    char m[3]={0}; if(fscanf(f,"%2s",m)!=1||strcmp(m,"P5")){fprintf(stderr,"%s: not P5\n",path);fclose(f);return NULL;}
    int v[3],got=0;
    while(got<3){ int c=fgetc(f);
        if(c=='#'){ while(c!='\n'&&c!=EOF)c=fgetc(f); continue; }
        if(c==' '||c=='\n'||c=='\t'||c=='\r')continue;
        ungetc(c,f); if(fscanf(f,"%d",&v[got])!=1){fclose(f);return NULL;} got++; }
    fgetc(f);
    *W=v[0];*H=v[1];
    size_t n=(size_t)v[0]*v[1];
    unsigned char *b=(unsigned char*)malloc(n);
    if(fread(b,1,n,f)!=n){fprintf(stderr,"%s: short read\n",path);free(b);fclose(f);return NULL;}
    fclose(f); return b;
}
static int write_pgm(const char *p,const unsigned char *d,int W,int H){
    FILE *f=fopen(p,"wb"); if(!f)return -1;
    fprintf(f,"P5\n%d %d\n255\n",W,H); fwrite(d,1,(size_t)W*H,f); fclose(f); return 0;
}

int main(int argc,char**argv){
    const char *lp="msl.pgm",*rp="msr.pgm",*op="disp_ms_dsp.pgm";
    int reps=1, nthreads=4, use_hvx=1, domain=3;
    const char *gexp=0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-l")&&i+1<argc)lp=argv[++i];
        else if(!strcmp(argv[i],"-r")&&i+1<argc)rp=argv[++i];
        else if(!strcmp(argv[i],"-o")&&i+1<argc)op=argv[++i];
        else if(!strcmp(argv[i],"-n")&&i+1<argc)reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-t")&&i+1<argc)nthreads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-x")&&i+1<argc)use_hvx=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-d")&&i+1<argc)domain=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-G")&&i+1<argc)gexp=argv[++i];
    }
    int W,H,W2,H2;
    unsigned char *L=read_pgm(lp,&W,&H), *R=read_pgm(rp,&W2,&H2);
    if(!L||!R) return 1;
    if(W!=W2||H!=H2){fprintf(stderr,"size mismatch\n");return 1;}
    int ow=W/2, oh=H/2;
    printf("CONFIG B multiscale  in %dx%d out %dx%d  domain=%d threads=%d hvx=%d reps=%d\n",
           W,H,ow,oh,domain,nthreads,use_hvx,reps);

    const char *domsuf = "&_dom=cdsp";
    switch(domain){
      case 0: domsuf="&_dom=adsp";  break;
      case 3: domsuf="&_dom=cdsp";  break;
      case 4: domsuf="&_dom=cdsp1"; break;
      case 5: domsuf="&_dom=gdsp0"; break;
      case 6: domsuf="&_dom=gdsp1"; break;
      default: fprintf(stderr,"unknown domain %d\n",domain); return 1;
    }
    char uri[512];
    snprintf(uri,sizeof uri,"%s%s",sgmms_URI,domsuf);

    struct remote_rpc_control_unsigned_module um;
    um.domain = domain; um.enable = 1;
    int uerr = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,(void*)&um,sizeof(um));
    printf("unsigned_pd enable: %s (0x%x)\n", uerr?"FAILED":"ok", uerr);

    remote_handle64 h=0;
    int nErr=sgmms_open(uri,&h);
    if(nErr){fprintf(stderr,"sgmms_open failed 0x%x uri=%s\n",nErr,uri);return 1;}
    uint64 need=0;
    if((nErr=sgmms_scratch_bytes(h,W,H,&need))){fprintf(stderr,"scratch_bytes 0x%x\n",nErr);return 1;}
    printf("scratch needed: %llu bytes (%.1f MB)\n",(unsigned long long)need,need/1048576.0);

    rpcmem_init();
    size_t wh=(size_t)W*H, owh=(size_t)ow*oh;
    unsigned char *dL=(unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,wh);
    unsigned char *dR=(unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,wh);
    unsigned char *dO=(unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,owh);
    unsigned char *dS=(unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,(size_t)need);
    if(!dL||!dR||!dO||!dS){fprintf(stderr,"rpcmem_alloc failed (scratch %.1f MB)\n",need/1048576.0);return 1;}
    memcpy(dL,L,wh); memcpy(dR,R,wh);

    double best=1e30, all[64]; uint64 dsp_cycles=0; uint64 ph[24]={0};
    if (reps > 64) reps = 64;
    for(int i=0;i<reps;i++){
        double t0=now_ms();
        nErr=sgmms_run(h,dL,(int)wh,dR,(int)wh,W,H,nthreads,use_hvx,
                       dS,(int)need,dO,(int)owh,&dsp_cycles,ph,24);
        double dt=now_ms()-t0;
        if(nErr){fprintf(stderr,"sgmms_run failed 0x%x\n",nErr);return 1;}
        printf("  rep %d: %.2f ms (dsp cycles %llu)\n",i,dt,(unsigned long long)dsp_cycles);
        all[i]=dt;
        if(dt<best)best=dt;
    }
    /* median of this invocation's reps (warm reps only if n >= 4) */
    int lo = (reps >= 4) ? 2 : 0;
    int nn = reps - lo;
    for(int i=lo;i<reps;i++) for(int j=i+1;j<reps;j++)
        if(all[j]<all[i]){double tt=all[i];all[i]=all[j];all[j]=tt;}
    double med = all[lo + nn/2];

    uint64_t hash=fnv1a64(dO,owh);
    /* WORK-PERFORMED MDE/s: both scales count. (W/2*H/2 + W*H) * 64 / t */
    double workpx=(double)ow*oh+(double)W*H;
    double mde=workpx*64.0/(med/1000.0);
    printf("\nRESULT CONFIG-B %dx%d->%dx%d D=64 paths=8 multiscale\n",W,H,ow,oh);
    printf("  median_ms    %.2f   (warm reps %d..%d of this invocation)\n",med,lo,reps-1);
    printf("  best_ms      %.2f\n",best);
    printf("  MDEs_work    %.0f   (= (%.3f+%.3f) Mpx x 64 / t)\n",mde,(double)ow*oh/1e6,(double)W*H/1e6);
    printf("  fps          %.2f\n",1000.0/med);
    printf("  fnv1a        %016llx\n",(unsigned long long)hash);
    printf("  dsp_pcycles  %llu\n",(unsigned long long)dsp_cycles);
    printf("  PHASES(cyc)  half %llu\n", (unsigned long long)ph[0]);
    printf("   s1: sobel %llu census %llu cost %llu agg %llu argmin %llu\n",
           (unsigned long long)ph[1],(unsigned long long)ph[2],(unsigned long long)ph[3],
           (unsigned long long)ph[4],(unsigned long long)ph[5]);
    printf("   s2: sobel %llu census %llu cost+merge %llu agg %llu argmin %llu\n",
           (unsigned long long)ph[6],(unsigned long long)ph[7],(unsigned long long)ph[8],
           (unsigned long long)ph[9],(unsigned long long)ph[10]);
    printf("   s2 diagonals within agg: %llu\n",(unsigned long long)ph[22]);
    printf("  VTCM    %s\n", ph[11]?"acquired":"NOT acquired (scratch fallback)");
    printf("  HW      num_workers=%llu num_hvx128_contexts=%llu\n",
           (unsigned long long)ph[14],(unsigned long long)ph[15]);
    printf("  LEFTOVER %llu pixels through the scalar fallback%s\n",
           (unsigned long long)ph[13], ph[13]?"   <<<< NOT A PURE VECTOR RUN":"");
    printf("  DSPMHZ  %.1f (pcycles/wall of last rep)\n",(double)dsp_cycles/(all[lo+nn-1]*1000.0));
    if(gexp){
        char got[32]; snprintf(got,sizeof got,"%016llx",(unsigned long long)hash);
        int ok = !strcasecmp(got,gexp);
        printf("  GOLDEN  expected %s got %s -> %s\n", gexp, got,
               ok?"MATCH":"*** MISMATCH -- THIS TIMING IS VOID ***");
        if(!ok){ write_pgm(op,dO,ow,oh); return 2; }
    } else {
        printf("  GOLDEN  no -G given -> THIS TIMING IS UNVERIFIED\n");
    }
    write_pgm(op,dO,ow,oh);
    printf("  wrote        %s\n",op);

    rpcmem_free(dL);rpcmem_free(dR);rpcmem_free(dO);rpcmem_free(dS);
    rpcmem_deinit(); sgmms_close(h);
    free(L);free(R);
    return 0;
}
