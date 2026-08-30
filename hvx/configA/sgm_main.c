/* sgm_main.c — ARM host for SGM on the Hexagon NSP.
 * Reads 95emulator's P5 PGMs, runs the DSP, hashes the output with the
 * SAME FNV-1a their harness uses so the result is directly comparable
 * to a golden that already agrees across six execution targets.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include "sgmdsp.h"
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

/* minimal P5 PGM reader (handles comments) */
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


/* ---- one full frame on one domain, in its own thread ---------------------- */
typedef struct {
    int domain, W, H, reps, nthreads, hvxthreads, use_hvx;
    const unsigned char *L, *R;
    double best_ms; uint64_t hash; int err; uint64 ph[16];
} frame_job_t;

static const char *dom_suffix(int d){
    switch(d){ case 0: return "&_dom=adsp"; case 3: return "&_dom=cdsp";
               case 4: return "&_dom=cdsp1"; case 5: return "&_dom=gdsp0";
               case 6: return "&_dom=gdsp1"; }
    return 0;
}

static void *frame_worker(void *arg)
{
    frame_job_t *j = (frame_job_t*)arg;
    char uri[512];
    const char *sfx = dom_suffix(j->domain);
    if(!sfx){ j->err = -1; return 0; }
    snprintf(uri,sizeof uri,"%s%s",sgmdsp_URI,sfx);
    struct remote_rpc_control_unsigned_module um;
    um.domain = j->domain; um.enable = 1;
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,(void*)&um,sizeof(um));
    remote_handle64 h=0;
    if((j->err = sgmdsp_open(uri,&h))) return 0;
    uint64 need=0;
    if((j->err = sgmdsp_scratch_bytes(h,j->W,j->H,&need))) return 0;
    size_t wh=(size_t)j->W*j->H;
    unsigned char *dL=(unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,wh);
    unsigned char *dR=(unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,wh);
    unsigned char *dO=(unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,wh);
    unsigned char *dS=(unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,(size_t)need);
    if(!dL||!dR||!dO||!dS){ j->err = -2; return 0; }
    memcpy(dL,j->L,wh); memcpy(dR,j->R,wh);
    uint64 cyc=0;
    j->best_ms=1e30;
    for(int i=0;i<j->reps;i++){
        double t0=now_ms();
        j->err = sgmdsp_run(h,dL,(int)wh,dR,(int)wh,j->W,j->H,
                            j->nthreads|(j->hvxthreads<<8),j->use_hvx,
                            dS,(int)need,dO,(int)wh,&cyc,j->ph,16);
        double dt=now_ms()-t0;
        if(j->err) return 0;
        if(dt<j->best_ms) j->best_ms=dt;
    }
    j->hash=fnv1a64(dO,wh);
    rpcmem_free(dL);rpcmem_free(dR);rpcmem_free(dO);rpcmem_free(dS);
    sgmdsp_close(h);
    return 0;
}

int main(int argc,char**argv){
    const char *lp="left.pgm",*rp="right.pgm",*op="disp_dsp.pgm";
    int reps=1, nthreads=1, use_hvx=0, domain=3, hvxthreads=0, ndom=0;
    const char *gexp=0;   /* -G <16 hex>: golden checked in the SAME run as the timing */
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-l")&&i+1<argc)lp=argv[++i];
        else if(!strcmp(argv[i],"-r")&&i+1<argc)rp=argv[++i];
        else if(!strcmp(argv[i],"-o")&&i+1<argc)op=argv[++i];
        else if(!strcmp(argv[i],"-n")&&i+1<argc)reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-t")&&i+1<argc)nthreads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-x")&&i+1<argc)use_hvx=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-d")&&i+1<argc)domain=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-T")&&i+1<argc)hvxthreads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-D")&&i+1<argc)ndom=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-G")&&i+1<argc)gexp=argv[++i];
    }
    int W,H,W2,H2;
    unsigned char *L=read_pgm(lp,&W,&H), *R=read_pgm(rp,&W2,&H2);
    if(!L||!R) return 1;
    if(W!=W2||H!=H2){fprintf(stderr,"size mismatch\n");return 1;}
    printf("image %dx%d  domain=%d threads=%d hvxthreads=%d hvx=%d reps=%d\n",
           W,H,domain,nthreads,hvxthreads?hvxthreads:nthreads,use_hvx,reps);

    if(ndom>=1){
        /* concurrent independent frames, one per NSP -- a THROUGHPUT figure.
         * Single-frame latency is untouched; SGM's 4 paths are global, so a
         * bit-exact latency split would need per-phase RPC with host barriers. */
        rpcmem_init();
        int doms[2]={domain,(domain==3)?4:3};
        frame_job_t jb[2]; pthread_t th[2];
        double t0=now_ms();
        for(int i=0;i<ndom && i<2;i++){
            memset(&jb[i],0,sizeof jb[i]);
            jb[i].domain=doms[i]; jb[i].W=W; jb[i].H=H; jb[i].reps=reps;
            jb[i].nthreads=nthreads; jb[i].hvxthreads=hvxthreads; jb[i].use_hvx=use_hvx;
            jb[i].L=L; jb[i].R=R;
            pthread_create(&th[i],0,frame_worker,&jb[i]);
        }
        for(int i=0;i<ndom && i<2;i++) pthread_join(th[i],0);
        double wall=now_ms()-t0;
        printf("\nDUAL-NSP THROUGHPUT (%d domains, %d reps each)\n",ndom,reps);
        for(int i=0;i<ndom && i<2;i++)
            printf("  dom %d: err=%d best_ms %.2f  fnv1a %016llx  %s\n",
                   jb[i].domain,jb[i].err,jb[i].best_ms,(unsigned long long)jb[i].hash,
                   jb[i].hash==0xb1b407b5949f0cc1ULL?"GOLDEN":"(check)");
        printf("  wall for %d frames: %.2f ms -> %.2f frames/s aggregate\n",
               ndom*reps, wall, (ndom*reps)/(wall/1000.0));
        rpcmem_deinit(); free(L); free(R);
        return 0;
    }
    char uri[512];
    if(ndom>=1){
        /* concurrent independent frames, one per NSP -- a THROUGHPUT figure.
         * Single-frame latency is untouched; SGM's 4 paths are global, so a
         * bit-exact latency split would need per-phase RPC with host barriers. */
        rpcmem_init();
        int doms[2]={domain,(domain==3)?4:3};
        frame_job_t jb[2]; pthread_t th[2];
        double t0=now_ms();
        for(int i=0;i<ndom && i<2;i++){
            memset(&jb[i],0,sizeof jb[i]);
            jb[i].domain=doms[i]; jb[i].W=W; jb[i].H=H; jb[i].reps=reps;
            jb[i].nthreads=nthreads; jb[i].hvxthreads=hvxthreads; jb[i].use_hvx=use_hvx;
            jb[i].L=L; jb[i].R=R;
            pthread_create(&th[i],0,frame_worker,&jb[i]);
        }
        for(int i=0;i<ndom && i<2;i++) pthread_join(th[i],0);
        double wall=now_ms()-t0;
        printf("\nDUAL-NSP THROUGHPUT (%d domains, %d reps each)\n",ndom,reps);
        for(int i=0;i<ndom && i<2;i++)
            printf("  dom %d: err=%d best_ms %.2f  fnv1a %016llx  %s\n",
                   jb[i].domain,jb[i].err,jb[i].best_ms,(unsigned long long)jb[i].hash,
                   jb[i].hash==0xb1b407b5949f0cc1ULL?"GOLDEN":"(check)");
        printf("  wall for %d frames: %.2f ms -> %.2f frames/s aggregate\n",
               ndom*reps, wall, (ndom*reps)/(wall/1000.0));
        rpcmem_deinit(); free(L); free(R);
        return 0;
    }
    const char *domsuf = "&_dom=cdsp";
    switch(domain){
      case 0: domsuf="&_dom=adsp";  break;
      case 3: domsuf="&_dom=cdsp";  break;
      case 4: domsuf="&_dom=cdsp1"; break;
      case 5: domsuf="&_dom=gdsp0"; break;
      case 6: domsuf="&_dom=gdsp1"; break;
      default: fprintf(stderr,"unknown domain %d\n",domain); return 1;
    }
    snprintf(uri,sizeof uri,"%s%s",sgmdsp_URI,domsuf);

    /* Unsigned PD: our skel is test-built, not Qualcomm-signed. Without this
     * the signed PD refuses it and open() returns 0x80000406 (module not found),
     * which looks identical to a missing file. */
    struct remote_rpc_control_unsigned_module um;
    um.domain = domain;   /* 0 adsp, 3 cdsp, 4 cdsp1, 5 gdsp0, 6 gdsp1 */
    um.enable = 1;
    int uerr = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,(void*)&um,sizeof(um));
    printf("unsigned_pd enable: %s (0x%x)\n", uerr?"FAILED":"ok", uerr);

    remote_handle64 h=0;
    int nErr=sgmdsp_open(uri,&h);
    if(nErr){fprintf(stderr,"sgmdsp_open failed 0x%x uri=%s\n",nErr,uri);return 1;}
    printf("DSP session open (%s)\n",uri);

    uint64 need=0;
    if((nErr=sgmdsp_scratch_bytes(h,W,H,&need))){fprintf(stderr,"scratch_bytes 0x%x\n",nErr);return 1;}
    printf("scratch needed: %llu bytes (%.1f MB)\n",(unsigned long long)need,need/1048576.0);

    rpcmem_init();
    size_t wh=(size_t)W*H;
    unsigned char *dL=(unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,wh);
    unsigned char *dR=(unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,wh);
    unsigned char *dO=(unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,wh);
    unsigned char *dS=(unsigned char*)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,(size_t)need);
    if(!dL||!dR||!dO||!dS){fprintf(stderr,"rpcmem_alloc failed (scratch %.1f MB)\n",need/1048576.0);return 1;}
    memcpy(dL,L,wh); memcpy(dR,R,wh);

    double best=1e30; uint64 dsp_cycles=0; uint64 ph[8]={0};
    for(int i=0;i<reps;i++){
        double t0=now_ms();
        nErr=sgmdsp_run(h,dL,(int)wh,dR,(int)wh,W,H,nthreads|(hvxthreads<<8),use_hvx,
                        dS,(int)need,dO,(int)wh,&dsp_cycles,ph,16);
        double dt=now_ms()-t0;
        if(nErr){fprintf(stderr,"sgmdsp_run failed 0x%x\n",nErr);return 1;}
        printf("  rep %d: %.2f ms (dsp cycles %llu)\n",i,dt,(unsigned long long)dsp_cycles);
        if(dt<best)best=dt;
    }

    if(use_hvx==9){
        double f=1458.1e6; double reg=(double)ph[3];
        printf("\nBWPROBE region=%.1f MB threads=%d\n",reg/1048576.0,nthreads);
        printf("  read   %llu cyc  %.2f GB/s\n",(unsigned long long)ph[0], reg/((double)ph[0]/f)/1e9);
        printf("  write  %llu cyc  %.2f GB/s\n",(unsigned long long)ph[1], reg/((double)ph[1]/f)/1e9);
        printf("  copy   %llu cyc  %.2f GB/s (2x traffic)\n",(unsigned long long)ph[2], 2*reg/((double)ph[2]/f)/1e9);
        printf("  HW num_workers=%llu num_hvx128_contexts=%llu\n",
               (unsigned long long)ph[4],(unsigned long long)ph[5]);
        return 0;
    }
    uint64_t hash=fnv1a64(dO,wh);
    unsigned long long Dv = ph[12] ? ph[12] : 64ULL;
    double mde=(double)W*H*(double)Dv/(best/1000.0);
    printf("\nRESULT %dx%d D=%llu paths=4 census9x7\n",W,H,Dv);
    printf("  best_ms      %.2f\n",best);
    printf("  MDE_per_s    %.0f\n",mde);
    printf("  Mpix_s       %.2f\n",(double)W*H/(best/1000.0)/1e6);
    printf("  fnv1a        %016llx\n",(unsigned long long)hash);
    printf("  dsp_pcycles  %llu\n",(unsigned long long)dsp_cycles);
    double tot=(double)(ph[0]+ph[1]+ph[2]+ph[3]); if(tot<1)tot=1;
    printf("  PHASES  census %.1f%%  cost %.1f%%  aggregate %.1f%%  argmin %.1f%%\n",
           100*ph[0]/tot,100*ph[1]/tot,100*ph[2]/tot,100*ph[3]/tot);
    printf("  HW      num_workers=%llu  num_hvx128_contexts=%llu\n",
           (unsigned long long)ph[4],(unsigned long long)ph[5]);
    printf("          (cycles: %llu / %llu / %llu / %llu)\n",
           (unsigned long long)ph[0],(unsigned long long)ph[1],
           (unsigned long long)ph[2],(unsigned long long)ph[3]);
    printf("  EXTRA   memset %llu  pathLR %llu  pathRL %llu  pathUD %llu  pathDU %llu\n",
           (unsigned long long)ph[6],(unsigned long long)ph[7],(unsigned long long)ph[8],
           (unsigned long long)ph[9],(unsigned long long)ph[10]);
    printf("  VTCM    %s\n", ph[11]?"acquired":"NOT acquired (scratch fallback)");
    printf("  PROLOG  %llu pcycles of the cost phase are the scalar first-%s-columns (%.1f%%)\n",
           (unsigned long long)ph[14], "DPAD", ph[1]?100.0*ph[14]/ph[1]:0.0);
    printf("  LEFTOVER %llu pixels through the scalar fallback%s\n",
           (unsigned long long)ph[13], ph[13]?"   <<<< NOT A PURE VECTOR RUN":"");
    if(gexp){
        char got[32]; snprintf(got,sizeof got,"%016llx",(unsigned long long)hash);
        int ok = !strcasecmp(got,gexp);
        printf("  GOLDEN  expected %s got %s -> %s\n", gexp, got, ok?"MATCH":"*** MISMATCH -- THIS TIMING IS VOID ***");
    } else {
        printf("  GOLDEN  no -G given -> THIS TIMING IS UNVERIFIED\n");
    }
    printf("  DSPMHZ  %.1f (pcycles/wall)\n", (double)dsp_cycles/(best*1000.0));
    write_pgm(op,dO,W,H);
    printf("  wrote        %s\n",op);

    rpcmem_free(dL);rpcmem_free(dR);rpcmem_free(dO);rpcmem_free(dS);
    rpcmem_deinit(); sgmdsp_close(h);
    free(L);free(R);
    return 0;
}
