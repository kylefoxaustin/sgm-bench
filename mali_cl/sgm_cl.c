/* sgm_cl.c — OpenCL host for the Mali SGM path. Plugs into the same harness,
 * so timing and golden comparison are identical to the CPU implementations.
 *
 * The whole pipeline is inside the timer per rule 2: census, the four
 * aggregation sweeps and the argmin. Buffer allocation and program build are
 * one-time and happen outside it (they are setup, not work).
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include "sgm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(v, what) do { if ((v) != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL %s failed: %d\n", (what), (int)(v)); return -1; } } while (0)

static cl_context   ctx;
static cl_command_queue q;
static cl_program   prog;
static cl_kernel    k_census, k_horiz, k_vert, k_argmin;
static cl_mem       b_l, b_r, b_cl, b_cr, b_S, b_disp;
static int          initW, initH, ready;

static char *slurp(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(s + 1); if (!b) { fclose(f); return NULL; }
    *n = fread(b, 1, s, f); b[*n] = 0; fclose(f); return b;
}

static int setup(int W, int H)
{
    cl_int e;
    cl_platform_id p; cl_uint np = 0;
    e = clGetPlatformIDs(1, &p, &np); CK(e, "GetPlatformIDs");
    cl_device_id d; cl_uint nd = 0;
    e = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, &nd); CK(e, "GetDeviceIDs");

    char nm[256] = {0}; size_t cu = 0;
    clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof nm, nm, 0);
    clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &cu, 0);
    fprintf(stderr, "# opencl device: %s (%u CUs)\n", nm, (unsigned)cu);

    ctx = clCreateContext(0, 1, &d, 0, 0, &e); CK(e, "CreateContext");
    q = clCreateCommandQueueWithProperties(ctx, d, 0, &e); CK(e, "CreateQueue");

    const char *path = getenv("SGM_CL_SRC");
    if (!path) path = "mali_cl/sgm.cl";
    size_t n = 0; char *src = slurp(path, &n);
    if (!src) { fprintf(stderr, "cannot read %s\n", path); return -1; }
    prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, &n, &e);
    CK(e, "CreateProgram");

    char opts[256];
    snprintf(opts, sizeof opts,
             "-DD_DISP=%d -DCW=%d -DCH=%d -DP1V=%d -DP2V=%d",
             SGM_D, SGM_CENSUS_W, SGM_CENSUS_H, SGM_P1, SGM_P2);
    e = clBuildProgram(prog, 1, &d, opts, 0, 0);
    if (e != CL_SUCCESS) {
        size_t ln = 0; clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, 0, 0, &ln);
        char *log = malloc(ln + 1);
        clGetProgramBuildInfo(prog, d, CL_PROGRAM_BUILD_LOG, ln, log, 0);
        log[ln] = 0; fprintf(stderr, "build log:\n%s\n", log);
        return -1;
    }
    free(src);
    k_census = clCreateKernel(prog, "census",   &e); CK(e, "kernel census");
    k_horiz  = clCreateKernel(prog, "agg_horiz",&e); CK(e, "kernel agg_horiz");
    k_vert   = clCreateKernel(prog, "agg_vert", &e); CK(e, "kernel agg_vert");
    k_argmin = clCreateKernel(prog, "argmin_k", &e); CK(e, "kernel argmin");

    const size_t N = (size_t)W * H;
    b_l  = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  N,        0, &e); CK(e, "buf l");
    b_r  = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  N,        0, &e); CK(e, "buf r");
    b_cl = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * 8,    0, &e); CK(e, "buf cl");
    b_cr = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * 8,    0, &e); CK(e, "buf cr");
    b_S  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * SGM_D * 2, 0, &e); CK(e, "buf S");
    b_disp = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, N,      0, &e); CK(e, "buf disp");
    initW = W; initH = H; ready = 1;
    return 0;
}

static int cl_run(const uint8_t *left, const uint8_t *right, int W, int H,
                  uint8_t *disp, int threads, sgm_stage_times *t)
{
    (void)threads;
    cl_int e;
    if (!ready || W != initW || H != initH) { if (setup(W, H)) return -1; }
    const size_t N = (size_t)W * H;

    double t0 = now_ms();
    e = clEnqueueWriteBuffer(q, b_l, CL_FALSE, 0, N, left,  0, 0, 0); CK(e, "wr l");
    e = clEnqueueWriteBuffer(q, b_r, CL_FALSE, 0, N, right, 0, 0, 0); CK(e, "wr r");
    cl_uchar zero = 0;
    e = clEnqueueFillBuffer(q, b_S, &zero, 1, 0, N * SGM_D * 2, 0, 0, 0); CK(e, "fill S");

    /* census on both images */
    size_t g2[2] = { ((size_t)W + 15) / 16 * 16, ((size_t)H + 3) / 4 * 4 };
    size_t l2[2] = { 16, 4 };
    for (int i = 0; i < 2; i++) {
        cl_mem in = i ? b_r : b_l, out = i ? b_cr : b_cl;
        clSetKernelArg(k_census, 0, sizeof in,  &in);
        clSetKernelArg(k_census, 1, sizeof out, &out);
        clSetKernelArg(k_census, 2, sizeof W, &W);
        clSetKernelArg(k_census, 3, sizeof H, &H);
        e = clEnqueueNDRangeKernel(q, k_census, 2, 0, g2, l2, 0, 0, 0);
        CK(e, "census");
    }
    clFinish(q);
    double t1 = now_ms();

    /* four sweeps: L, R (per row) then U, D (per column) */
    const size_t WI = SGM_D / 4;
    struct { cl_kernel k; size_t groups; int dir; } sw[4] = {
        { k_horiz, (size_t)H, +1 }, { k_horiz, (size_t)H, -1 },
        { k_vert,  (size_t)W, +1 }, { k_vert,  (size_t)W, -1 },
    };
    for (int i = 0; i < 4 && i < SGM_PATHS; i++) {
        size_t g = sw[i].groups * WI, l = WI;
        clSetKernelArg(sw[i].k, 0, sizeof b_cl, &b_cl);
        clSetKernelArg(sw[i].k, 1, sizeof b_cr, &b_cr);
        clSetKernelArg(sw[i].k, 2, sizeof b_S,  &b_S);
        clSetKernelArg(sw[i].k, 3, sizeof W, &W);
        clSetKernelArg(sw[i].k, 4, sizeof H, &H);
        clSetKernelArg(sw[i].k, 5, sizeof sw[i].dir, &sw[i].dir);
        e = clEnqueueNDRangeKernel(q, sw[i].k, 1, 0, &g, &l, 0, 0, 0);
        CK(e, "sweep");
        clFinish(q);
    }
    double t2 = now_ms();

    clSetKernelArg(k_argmin, 0, sizeof b_S, &b_S);
    clSetKernelArg(k_argmin, 1, sizeof b_disp, &b_disp);
    clSetKernelArg(k_argmin, 2, sizeof W, &W);
    clSetKernelArg(k_argmin, 3, sizeof H, &H);
    e = clEnqueueNDRangeKernel(q, k_argmin, 2, 0, g2, l2, 0, 0, 0); CK(e, "argmin");
    e = clEnqueueReadBuffer(q, b_disp, CL_TRUE, 0, N, disp, 0, 0, 0); CK(e, "rd disp");
    double t3 = now_ms();

    if (t) { t->census_ms = t1 - t0; t->cost_ms = -1;
             t->aggregate_ms = t2 - t1; t->argmin_ms = t3 - t2; }
    return 0;
}

const sgm_impl SGM_IMPL = { "mali_cl", cl_run };
