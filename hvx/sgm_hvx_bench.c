/* sgm_hvx_bench.c — HVX v73 twin of colmajor's vertical-path aggregation.
 *
 * Measures the term 95emulator's cost model says decides the layout:
 * per-d-iteration STATE TRAFFIC vs LANE UTILISATION.
 *
 * Column-major puts COLUMNS in the vector lanes, so min-over-d needs no
 * cross-lane reduce and d±1 are array indices, not vector shifts. The price
 * is state: L[d][lane] is D*BW bytes and cannot live in registers.
 *
 * Two kernels, identical arithmetic:
 *   NAIVE   — reloads L[d-1], L[d], L[d+1] every iteration   (3 loads/d)
 *   ROLLING — sweeps d keeping prev/cur/next in registers    (1 load/d)
 * The delta between them IS the state-traffic term, measured.
 *
 * Correctness is gated in the SAME run as the timing: a scalar reference
 * computes the identical recurrence and every output byte must match.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define SGM_D   64
#ifndef BW
#define BW      128
#endif          /* columns per block; multiple of 128 */
#ifndef ROWS
#define ROWS    64
#endif        /* rows; scaled so ROWS*BW is constant */

#define P1 20
#define P2 192
#define SPL(b) Q6_V_vsplat_R((int)(((uint32_t)(b))*0x01010101u))

static uint8_t Lbuf [SGM_D*BW] __attribute__((aligned(128)));
static uint8_t Obuf [SGM_D*BW] __attribute__((aligned(128)));
static uint8_t Cbuf [SGM_D*BW] __attribute__((aligned(128)));
static uint8_t MNbuf[BW]       __attribute__((aligned(128)));
static uint8_t Ref  [SGM_D*BW] __attribute__((aligned(128)));

/* ---- min over d, per column: D vector mins, NO cross-lane reduce ---- */
static void colmin_hvx(const uint8_t *L, uint8_t *mn)
{
    for (int v = 0; v < BW; v += 128) {
        HVX_Vector m = *(const HVX_Vector *)(L + v);
        for (int d = 1; d < SGM_D; d++)
            m = Q6_Vub_vmin_VubVub(m, *(const HVX_Vector *)(L + (size_t)d*BW + v));
        *(HVX_Vector *)(mn + v) = m;
    }
}

/* ---- NAIVE: three loads of L per d-iteration ---- */
static void step_naive(const uint8_t *L, const uint8_t *C, uint8_t *out, const uint8_t *mn)
{
    const HVX_Vector p1 = SPL(P1), p2 = SPL(P2), big = SPL(255);
    for (int v = 0; v < BW; v += 128) {
    const HVX_Vector mnv = *(const HVX_Vector *)(mn + v);
    const HVX_Vector p2t = Q6_Vub_vadd_VubVub_sat(mnv, p2);
    for (int d = 0; d < SGM_D; d++) {
        const uint8_t *Ld = L + (size_t)d*BW + v;
        HVX_Vector cur = *(const HVX_Vector *)Ld;
        HVX_Vector lo  = (d == 0)         ? big : *(const HVX_Vector *)(Ld - BW);
        HVX_Vector hi  = (d == SGM_D - 1) ? big : *(const HVX_Vector *)(Ld + BW);
        HVX_Vector m = Q6_Vub_vmin_VubVub(cur, Q6_Vub_vadd_VubVub_sat(lo, p1));
        m = Q6_Vub_vmin_VubVub(m, Q6_Vub_vadd_VubVub_sat(hi, p1));
        m = Q6_Vub_vmin_VubVub(m, p2t);
        *(HVX_Vector *)(out + (size_t)d*BW + v) =
            Q6_Vub_vadd_VubVub_sat(*(const HVX_Vector *)(C + (size_t)d*BW + v),
                                   Q6_Vb_vsub_VbVb(m, mnv));
    }}
}

/* ---- ROLLING: one load of L per d-iteration, prev/cur/next in registers ---- */
static void step_rolling(const uint8_t *L, const uint8_t *C, uint8_t *out, const uint8_t *mn)
{
    const HVX_Vector p1 = SPL(P1), p2 = SPL(P2), big = SPL(255);
    for (int v = 0; v < BW; v += 128) {
    const HVX_Vector mnv2 = *(const HVX_Vector *)(mn + v);
    const HVX_Vector p2t2 = Q6_Vub_vadd_VubVub_sat(mnv2, p2);
    HVX_Vector lo  = big;
    HVX_Vector cur = *(const HVX_Vector *)(L + v);
    for (int d = 0; d < SGM_D; d++) {
        HVX_Vector hi = (d == SGM_D - 1) ? big : *(const HVX_Vector *)(L + (size_t)(d+1)*BW + v);
        HVX_Vector m = Q6_Vub_vmin_VubVub(cur, Q6_Vub_vadd_VubVub_sat(lo, p1));
        m = Q6_Vub_vmin_VubVub(m, Q6_Vub_vadd_VubVub_sat(hi, p1));
        m = Q6_Vub_vmin_VubVub(m, p2t2);
        *(HVX_Vector *)(out + (size_t)d*BW + v) =
            Q6_Vub_vadd_VubVub_sat(*(const HVX_Vector *)(C + (size_t)d*BW + v),
                                   Q6_Vb_vsub_VbVb(m, mnv2));
        lo = cur; cur = hi;
    }}
}

/* ---- scalar reference: the SAME recurrence, no vectors ---- */
static uint8_t sat_add(uint8_t a, uint8_t b){ unsigned s=(unsigned)a+b; return s>255?255:(uint8_t)s; }
static void step_ref(const uint8_t *L, const uint8_t *C, uint8_t *out, const uint8_t *mn)
{
    for (int d = 0; d < SGM_D; d++)
        for (int x = 0; x < BW; x++) {
            uint8_t cur = L[(size_t)d*BW+x];
            uint8_t lo  = (d==0)         ? 255 : L[(size_t)(d-1)*BW+x];
            uint8_t hi  = (d==SGM_D-1)   ? 255 : L[(size_t)(d+1)*BW+x];
            uint8_t mnv = mn[x];
            uint8_t m = cur;
            uint8_t t;
            t = sat_add(lo,P1);  if (t<m) m=t;
            t = sat_add(hi,P1);  if (t<m) m=t;
            t = sat_add(mnv,P2); if (t<m) m=t;
            out[(size_t)d*BW+x] = sat_add(C[(size_t)d*BW+x], (uint8_t)(m - mnv));
        }
}

static void fill(uint32_t seed)
{
    uint32_t s = seed;
    for (int i = 0; i < SGM_D*BW; i++) { s = s*1664525u+1013904223u; Lbuf[i] = (s>>16)&0x3F; }
    for (int i = 0; i < SGM_D*BW; i++) { s = s*1664525u+1013904223u; Cbuf[i] = (s>>16)%63; }
}

int main(int argc, char **argv)
{
    /* One ARM per simulator run; the sim's own Pcycle total is the clock.
     * arm 0 = baseline (gate only, no timed loop) -> subtract it. */
    int arm = (argc > 1) ? atoi(argv[1]) : 0;

    fill(1);
    colmin_hvx(Lbuf, MNbuf);

    /* ---- CORRECTNESS GATE, same run as the timing, every arm ---- */
    step_ref(Lbuf, Cbuf, Ref, MNbuf);
    step_naive(Lbuf, Cbuf, Obuf, MNbuf);
    int bad_n = memcmp(Obuf, Ref, sizeof Ref) ? 1 : 0;
    memset(Obuf, 0, sizeof Obuf);
    step_rolling(Lbuf, Cbuf, Obuf, MNbuf);
    int bad_r = memcmp(Obuf, Ref, sizeof Ref) ? 1 : 0;
    printf("CORRECTNESS naive=%s rolling=%s\n", bad_n?"FAIL":"OK", bad_r?"FAIL":"OK");
    if (bad_n || bad_r) { printf("ABORT: kernel wrong, timing meaningless\n"); return 2; }

    if (arm == 1)
        for (int y = 0; y < ROWS; y++) { colmin_hvx(Lbuf, MNbuf); step_naive(Lbuf, Cbuf, Obuf, MNbuf); }
    else if (arm == 2)
        for (int y = 0; y < ROWS; y++) { colmin_hvx(Lbuf, MNbuf); step_rolling(Lbuf, Cbuf, Obuf, MNbuf); }

    printf("ARM %d DONE ROWS=%d BW=%d D=%d checksum=%u\n",
           arm, ROWS, BW, SGM_D, (unsigned)Obuf[7] + Obuf[1000]);
    return 0;
}
