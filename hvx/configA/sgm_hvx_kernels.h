/* sgm_hvx_kernels.h — the D-generic HVX primitives for SGM on Hexagon v73.
 *
 * Included by BOTH sgmdsp_imp.c (ships) and sim/gen.c (validates), so what the
 * simulator checks is byte-for-byte what runs on the board.
 *
 * GEOMETRY.  D is padded up to DPAD, the next power of two >= D (32/64/128).
 * A 128-byte HVX vector holds PPV = 128/DPAD pixels, one per DPAD-lane
 * segment.  Lanes [D, DPAD) of every segment are "pad" and permanently hold
 * 255, which buys three things:
 *   - a pad lane can never win a min: real L <= 62 + P2 = 254 < 255, always;
 *   - lane D-1's d+1 neighbour reads 255, which IS the reference's "d+1 out of
 *     range = 255" rule, so no extra masking in the inner loop;
 *   - real S <= 4*254 = 1016 while pad S = 4*255 = 1020, so a pad lane cannot
 *     win the argmin either, and there is no tie to break.
 * The min-over-d reduce must stay INSIDE a segment.  Q6_V_vdelta_VV(v,splat(k))
 * is a butterfly -- lane i reads lane i^k -- so keeping k < DPAD keeps every
 * reduce inside its own DPAD-aligned segment.  That is what makes PPV>1 legal.
 * (sim/prim.c verifies the butterfly for k = 1..64 and the reversal controls.)
 */
#ifndef SGM_HVX_KERNELS_H
#define SGM_HVX_KERNELS_H

#ifndef SGM_D
#define SGM_D 64
#endif
#if   SGM_D <= 32
#define DPAD 32
#elif SGM_D <= 64
#define DPAD 64
#elif SGM_D <= 128
#define DPAD 128
#elif SGM_D <= 256
#define DPAD 256
#else
#error "SGM_D > 256 is not supported by this build"
#endif
#if (SGM_D % 16) != 0
#error "SGM_D must be a multiple of 16"
#endif

/* Two regimes.  D <= 128: one vector holds PPV = 128/DPAD whole pixels.
 * D > 128: one PIXEL needs VPC = DPAD/128 vectors, and PPV collapses to 1. */
#if DPAD <= 128
#define PPV  (128/DPAD)     /* pixels per 128-byte vector                    */
#define VPC  1              /* vectors per pixel                            */
#define SHF  (128-DPAD)     /* valign offset that assembles PPV rows         */
#else
#define PPV  1
#define VPC  (DPAD/128)
#define SHF  0
#endif
#define NVP  (DPAD/16)      /* census-word vectors per pixel in build_cost   */
#if PPV <= 2
#define HB 1                /* x-steps batched so every S store is a full    */
#else
#define HB 2                /* aligned vector (PPV=4 needs two)              */
#endif

#ifndef P1
#define P1 20
#endif
#ifndef P2
#define P2 192
#endif
#ifndef COST_INVALID
#define COST_INVALID 62
#endif
#ifndef SGM_L2PF
#define SGM_L2PF(a,s,w,h) do{}while(0)
#endif

#define SPL(b) Q6_V_vsplat_R((int)(((unsigned)(b))*0x01010101u))

typedef struct { HVX_Vector mA, mB, pad, p1, p2, big; } pk_t;

/* Built from an explicit byte table: unambiguous at every DPAD, and it runs
 * once per job, never per pixel. */
static pk_t pk_init(void)
{
    unsigned char a[128] __attribute__((aligned(128)));
    unsigned char b[128] __attribute__((aligned(128)));
    unsigned char c[128] __attribute__((aligned(128)));
    for (int i = 0; i < 128; i++) {
        int s = i & (DPAD-1);
        a[i] = (s == 0 && i > 0) ? 255 : 0;   /* d-1 of a segment's d=0 */
        b[i] = (s == DPAD-1)     ? 255 : 0;   /* d+1 of a segment's top */
        c[i] = (s >= SGM_D)      ? 255 : 0;   /* pad lanes              */
    }
    pk_t k;
    k.mA = *(const HVX_Vector *)a;
    k.mB = *(const HVX_Vector *)b;
    k.pad= *(const HVX_Vector *)c;
    k.p1 = SPL(P1); k.p2 = SPL(P2); k.big = SPL(255);
    return k;
}

#if VPC == 1
/* One aggregation step for PPV pixels at once.  C must already carry 255 in
 * its pad lanes -- build_cost writes them there. */
static inline HVX_Vector step2(HVX_Vector Lp, HVX_Vector C, const pk_t *k)
{
    HVX_Vector mn = Lp;
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(1)));
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(2)));
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(4)));
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(8)));
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(16)));
#if DPAD >= 64
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(32)));
#endif
#if DPAD >= 128
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(64)));
#endif
    HVX_Vector lo = Q6_V_vor_VV(Q6_V_valign_VVR(Lp, k->big, 127), k->mA);
    HVX_Vector hi = Q6_V_vor_VV(Q6_V_valign_VVR(k->big, Lp, 1),   k->mB);
    HVX_Vector m  = Q6_Vub_vmin_VubVub(Lp, Q6_Vub_vadd_VubVub_sat(lo, k->p1));
    m = Q6_Vub_vmin_VubVub(m, Q6_Vub_vadd_VubVub_sat(hi, k->p1));
    m = Q6_Vub_vmin_VubVub(m, Q6_Vub_vadd_VubVub_sat(mn, k->p2));
    return Q6_Vub_vadd_VubVub_sat(C, Q6_Vb_vsub_VbVb(m, mn));
}

#endif /* VPC == 1 */

/* uint8 -> uint16, IN ORDER.  Q6_Wuh_vzxt_Vub alone is deinterleaved (its low
 * vector holds the EVEN bytes); the vshuff puts both halves back in order. */
static inline HVX_VectorPair widen_pair(HVX_Vector L)
{
    HVX_VectorPair p = Q6_Wuh_vzxt_Vub(L);
    return Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -2);
}
static inline HVX_VectorPair widen_add_pair(HVX_Vector a, HVX_Vector b)
{
    HVX_VectorPair p = Q6_Wh_vadd_VubVub(a, b);
    return Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -2);
}

#if VPC == 1
/* Vertical sweep store: the PPV pixels are adjacent COLUMNS, so their S
 * records are contiguous -- 2 aligned vectors, at every PPV. */
static inline void acc2(HVX_Vector L, HVX_Vector *Sp, int first)
{
    HVX_VectorPair q = widen_pair(L);
    HVX_Vector a = Q6_V_lo_W(q), b = Q6_V_hi_W(q);
    Sp[0] = first ? a : Q6_Vh_vadd_VhVh(Sp[0], a);
    Sp[1] = first ? b : Q6_Vh_vadd_VhVh(Sp[1], b);
}

/* ---- horizontal sweep: PPV adjacent ROWS in one vector ---------------------
 * Row y+i's pixel must land in lanes [i*DPAD, (i+1)*DPAD).  Loading row i at
 * (address - SHF) puts its DPAD bytes in the TOP DPAD lanes, and
 * valign(acc, that, SHF) = that[SHF..127] ++ acc[0..SHF-1] shifts the running
 * assembly down by one segment.  CV carries a 128-byte front pad so the
 * negative offset is legal at x=0,y=0.
 */
static inline HVX_Vector hz_cost(const unsigned char * const *A, size_t o)
{
    HVX_Vector acc = *(const HVX_UVector *)(A[PPV-1] + o);
#if PPV >= 2
    acc = Q6_V_valign_VVR(acc, *(const HVX_UVector *)(A[PPV-2] + o - SHF), SHF);
#endif
#if PPV >= 4
    acc = Q6_V_valign_VVR(acc, *(const HVX_UVector *)(A[1] + o - SHF), SHF);
    acc = Q6_V_valign_VVR(acc, *(const HVX_UVector *)(A[0] + o - SHF), SHF);
#endif
    return acc;
}

/* Scatter HB x-steps' widened contributions to the PPV row bases.
 * o2 = byte offset of the FIRST x of the batch inside a row's S array.
 * At PPV=4 one row needs only 64 bytes per pixel, so two x are transposed
 * into one aligned vector per row with a single vshuff each (block exchange
 * at 64-byte granularity -- verified in sim/prim.c). */
static inline void hz_store(unsigned short * const *Sr, size_t o2,
                            const HVX_VectorPair *w, int first)
{
#if PPV == 1
    HVX_Vector *p = (HVX_Vector *)((unsigned char *)Sr[0] + o2);
    HVX_Vector a = Q6_V_lo_W(w[0]), b = Q6_V_hi_W(w[0]);
    p[0] = first ? a : Q6_Vh_vadd_VhVh(p[0], a);
    p[1] = first ? b : Q6_Vh_vadd_VhVh(p[1], b);
#elif PPV == 2
    HVX_Vector *p0 = (HVX_Vector *)((unsigned char *)Sr[0] + o2);
    HVX_Vector *p1 = (HVX_Vector *)((unsigned char *)Sr[1] + o2);
    HVX_Vector a = Q6_V_lo_W(w[0]), b = Q6_V_hi_W(w[0]);
    *p0 = first ? a : Q6_Vh_vadd_VhVh(*p0, a);
    *p1 = first ? b : Q6_Vh_vadd_VhVh(*p1, b);
#else
    HVX_VectorPair q01 = Q6_W_vshuff_VVR(Q6_V_lo_W(w[1]), Q6_V_lo_W(w[0]), -64);
    HVX_VectorPair q23 = Q6_W_vshuff_VVR(Q6_V_hi_W(w[1]), Q6_V_hi_W(w[0]), -64);
    HVX_Vector v0 = Q6_V_lo_W(q01), v1 = Q6_V_hi_W(q01);
    HVX_Vector v2 = Q6_V_lo_W(q23), v3 = Q6_V_hi_W(q23);
    HVX_Vector *p0 = (HVX_Vector *)((unsigned char *)Sr[0] + o2);
    HVX_Vector *p1 = (HVX_Vector *)((unsigned char *)Sr[1] + o2);
    HVX_Vector *p2 = (HVX_Vector *)((unsigned char *)Sr[2] + o2);
    HVX_Vector *p3 = (HVX_Vector *)((unsigned char *)Sr[3] + o2);
    *p0 = first ? v0 : Q6_Vh_vadd_VhVh(*p0, v0);
    *p1 = first ? v1 : Q6_Vh_vadd_VhVh(*p1, v1);
    *p2 = first ? v2 : Q6_Vh_vadd_VhVh(*p2, v2);
    *p3 = first ? v3 : Q6_Vh_vadd_VhVh(*p3, v3);
#endif
}

#endif /* VPC == 1 */

/* The first XS = DPAD columns of a row are the only ones where d can exceed x,
 * and the reference wants COST_INVALID there.  Doing them scalar costs 10% of
 * the cost phase at D=64 and 46% at D=240 (measured, phases[14]).  Instead run
 * the same vector kernel -- the loads simply reach back into the previous row's
 * census words, which is readable memory whose value is then thrown away -- and
 * overwrite the out-of-range lanes with one compare and one mux.
 * lane i of segment p is valid iff d(i) <= x + p. */
static inline HVX_Vector lane_dindex(int hi128)
{
    unsigned char t[128] __attribute__((aligned(128)));
    for (int i = 0; i < 128; i++) t[i] = (unsigned char)((i & (DPAD-1)) + (hi128 ? 128 : 0));
    return *(const HVX_Vector *)t;
}
static inline HVX_Vector lane_poffset(void)
{
    unsigned char t[128] __attribute__((aligned(128)));
    for (int i = 0; i < 128; i++) t[i] = (unsigned char)(i / DPAD);
    return *(const HVX_Vector *)t;
}

/* ---- build_cost ------------------------------------------------------------
 * For one pixel, cr[x-DPAD+1 .. x] is DPAD consecutive 8-byte census words =
 * NVP HVX vectors.  xor against a splatted cl[x], popcount per halfword, then
 * vdmpy + vror fold each 8-byte word's bits into one word lane.  Three vpacke
 * stages compact eight such vectors' 16 useful bytes each into 128 contiguous
 * bytes in input order -- but with d REVERSED inside each pixel, because entry
 * e sits at address x-DPAD+1+e, i.e. d = DPAD-1-e.  One vdelta with
 * splat(DPAD-1) undoes that inside every DPAD-aligned segment.
 */
static inline HVX_Vector splat8(unsigned long long v)
{
    return Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R((int)(unsigned)(v >> 32)),
                                     Q6_V_vsplat_R((int)(unsigned)v), -4));
}
static inline HVX_Vector cost_rvec(const unsigned char *b, int k, HVX_Vector clv)
{
    HVX_Vector t = Q6_Vw_vdmpy_VhRb(Q6_Vh_vpopcount_Vh(
                       Q6_V_vxor_VV(*(const HVX_UVector *)(b + (k)*128), clv)), 0x01010101);
    return Q6_Vw_vadd_VwVw(t, Q6_V_vror_VR(t, 4));
}
/* Eight separate arguments, not an array: an array of HVX_Vector spills to
 * memory and cost 43.7M -> 66.1M cycles the first time I wrote it that way. */
static inline HVX_Vector cost_pack8(HVX_Vector r0, HVX_Vector r1, HVX_Vector r2, HVX_Vector r3,
                                    HVX_Vector r4, HVX_Vector r5, HVX_Vector r6, HVX_Vector r7)
{
    HVX_Vector s0 = Q6_Vb_vpacke_VhVh(r1, r0);
    HVX_Vector s1 = Q6_Vb_vpacke_VhVh(r3, r2);
    HVX_Vector s2 = Q6_Vb_vpacke_VhVh(r5, r4);
    HVX_Vector s3 = Q6_Vb_vpacke_VhVh(r7, r6);
    HVX_Vector t0 = Q6_Vb_vpacke_VhVh(s1, s0);
    HVX_Vector t1 = Q6_Vb_vpacke_VhVh(s3, s2);
    return Q6_Vb_vpacke_VhVh(t1, t0);
}
#if VPC == 1
static inline HVX_Vector cost_group(const unsigned long long *clr,
                                    const unsigned long long *crr,
                                    int x, HVX_Vector padv)
{
#define CB(p) ((const unsigned char *)(crr + (x + (p)) - (DPAD-1)))
#if PPV == 4                                  /* DPAD=32,  NVP=2 */
    HVX_Vector c0=splat8(clr[x]),  c1=splat8(clr[x+1]);
    HVX_Vector c2=splat8(clr[x+2]),c3=splat8(clr[x+3]);
    HVX_Vector v = cost_pack8(cost_rvec(CB(0),0,c0), cost_rvec(CB(0),1,c0),
                              cost_rvec(CB(1),0,c1), cost_rvec(CB(1),1,c1),
                              cost_rvec(CB(2),0,c2), cost_rvec(CB(2),1,c2),
                              cost_rvec(CB(3),0,c3), cost_rvec(CB(3),1,c3));
#elif PPV == 2                                /* DPAD=64,  NVP=4 */
    HVX_Vector c0=splat8(clr[x]), c1=splat8(clr[x+1]);
    HVX_Vector v = cost_pack8(cost_rvec(CB(0),0,c0), cost_rvec(CB(0),1,c0),
                              cost_rvec(CB(0),2,c0), cost_rvec(CB(0),3,c0),
                              cost_rvec(CB(1),0,c1), cost_rvec(CB(1),1,c1),
                              cost_rvec(CB(1),2,c1), cost_rvec(CB(1),3,c1));
#else                                         /* DPAD=128, NVP=8 */
    HVX_Vector c0=splat8(clr[x]);
    HVX_Vector v = cost_pack8(cost_rvec(CB(0),0,c0), cost_rvec(CB(0),1,c0),
                              cost_rvec(CB(0),2,c0), cost_rvec(CB(0),3,c0),
                              cost_rvec(CB(0),4,c0), cost_rvec(CB(0),5,c0),
                              cost_rvec(CB(0),6,c0), cost_rvec(CB(0),7,c0));
#endif
#undef CB
    return Q6_V_vor_VV(Q6_V_vdelta_VV(v, SPL(DPAD-1)), padv);
}
/* Same, but with the x < DPAD lanes forced to COST_INVALID.  The mask must be
 * applied BEFORE the pad OR: a pad lane has d >= SGM_D, so it is "out of range"
 * by this test and would otherwise come out 62 instead of 255. */
static inline HVX_Vector cost_group_edge(const unsigned long long *clr,
                                         const unsigned long long *crr, int x,
                                         HVX_Vector padv, HVX_Vector dv, HVX_Vector pv)
{
    HVX_Vector v = cost_group(clr, crr, x, Q6_V_vzero());
    HVX_Vector t = Q6_Vb_vadd_VbVb(SPL(x), pv);
    v = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VubVub(dv, t), SPL(COST_INVALID), v);
    return Q6_V_vor_VV(v, padv);
}

#endif /* VPC == 1 */

/* ---- argmin ----------------------------------------------------------------
 * KEY = (S << log2(KSEG)) | d turns "lowest d wins ties" into a plain unsigned
 * min.  S <= 4*254 = 1016, so the key fits a uint16 while the group is <= 64
 * halfwords.  At DPAD = 128 a 7-bit d would overflow (1016*128 > 65535), so
 * that case reduces the two 64-lane halves separately with a 6-bit key and
 * then compares the two S values explicitly -- which is also what keeps
 * "lowest d wins" correct ACROSS the halves (a tie must go to the low half).
 */
#if DPAD < 64
#define KSEG DPAD
#else
#define KSEG 64
#endif
#define KSH  (KSEG == 32 ? 5 : 6)
#if DPAD <= 64
#define AMIN_PPS (64/DPAD)      /* pixels consumed per S vector */
#else
#define AMIN_PPS 1
#endif

static inline HVX_Vector amin_key(const unsigned short *sp, HVX_Vector idx)
{
    HVX_Vector k = Q6_V_vor_VV(Q6_Vh_vasl_VhR(*(const HVX_Vector *)sp, KSH), idx);
#if KSEG == 64
    /* the group IS the whole vector, so vror works and -- unlike vdelta -- needs
     * no live vector control register per step (that cost 19.0M -> 55.9M cycles) */
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vror_VR(k,  2));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vror_VR(k,  4));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vror_VR(k,  8));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vror_VR(k, 16));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vror_VR(k, 32));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vror_VR(k, 64));
#else
    /* KSEG < 64: the reduce must stay inside each 2*KSEG-byte group, and vror
     * wraps the whole 128 bytes, so the butterfly is the only option. */
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vdelta_VV(k, SPL(2)));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vdelta_VV(k, SPL(4)));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vdelta_VV(k, SPL(8)));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vdelta_VV(k, SPL(16)));
    k = Q6_Vuh_vmin_VuhVuh(k, Q6_V_vdelta_VV(k, SPL(32)));
#endif
    return k;
}
/* consume AMIN_PPS pixels at q, push their answer into the accumulator */
#if DPAD >= 128
/* one 64-lane sub-vector -> (S, d-within-sub) as two halfword vectors */
static inline void amin_split(const unsigned short *q, HVX_Vector idx, HVX_Vector mk,
                              int base, HVX_Vector *Sv, HVX_Vector *Dv)
{
    HVX_Vector k = amin_key(q, idx);
    *Sv = Q6_Vuh_vlsr_VuhR(k, KSH);
    *Dv = Q6_Vh_vadd_VhVh(Q6_V_vand_VV(k, mk),
                          Q6_V_vsplat_R((int)((unsigned)base*0x00010001u)));
}
/* pick the lower S; on a tie keep the FIRST argument (lower d) */
static inline void amin_pick(HVX_Vector Sa, HVX_Vector Da, HVX_Vector Sb, HVX_Vector Db,
                             HVX_Vector *So, HVX_Vector *Do)
{
    HVX_VectorPred g = Q6_Q_vcmp_gt_VuhVuh(Sa, Sb);
    *So = Q6_V_vmux_QVV(g, Sb, Sa);
    *Do = Q6_V_vmux_QVV(g, Db, Da);
}
#endif

static inline HVX_Vector amin_push(HVX_Vector a, const unsigned short *q,
                                   HVX_Vector idx, HVX_Vector mk)
{
#if DPAD == 256
    HVX_Vector S0,D0,S1,D1,S2,D2,S3,D3, SA,DA,SB,DB, SC,DC;
    amin_split(q,       idx, mk,   0, &S0,&D0);
    amin_split(q +  64, idx, mk,  64, &S1,&D1);
    amin_split(q + 128, idx, mk, 128, &S2,&D2);
    amin_split(q + 192, idx, mk, 192, &S3,&D3);
    amin_pick(S0,D0,S1,D1,&SA,&DA);
    amin_pick(S2,D2,S3,D3,&SB,&DB);
    amin_pick(SA,DA,SB,DB,&SC,&DC);
    return Q6_V_valign_VVR(DC, a, 2);
#elif DPAD == 128
    HVX_Vector klo = amin_key(q,      idx);
    HVX_Vector khi = amin_key(q + 64, idx);
    HVX_VectorPred sel = Q6_Q_vcmp_gt_VuhVuh(Q6_Vuh_vlsr_VuhR(klo, KSH),
                                             Q6_Vuh_vlsr_VuhR(khi, KSH));
    HVX_Vector dv = Q6_V_vmux_QVV(sel,
                        Q6_Vh_vadd_VhVh(Q6_V_vand_VV(khi, mk), Q6_V_vsplat_R(0x00400040)),
                        Q6_V_vand_VV(klo, mk));
    return Q6_V_valign_VVR(dv, a, 2);
#elif DPAD == 64
    return Q6_V_valign_VVR(amin_key(q, idx), a, 2);
#else
    HVX_Vector k = amin_key(q, idx);
    a = Q6_V_valign_VVR(k, a, 2);
    return Q6_V_valign_VVR(Q6_V_vror_VR(k, 64), a, 2);
#endif
}
#if DPAD >= 128
#define AMIN_FIN(v, mk) (v)                       /* already an index */
#else
#define AMIN_FIN(v, mk) Q6_V_vand_VV((v), (mk))
#endif

static void argmin_hvx(const unsigned short *S, unsigned char *disp,
                       unsigned long long p0, unsigned long long p1)
{
    unsigned short it[64] __attribute__((aligned(128)));
    for (int i = 0; i < 64; i++) it[i] = (unsigned short)(i & (KSEG-1));
    const HVX_Vector idx = *(const HVX_Vector *)it;
    const HVX_Vector mk  = Q6_V_vsplat_R((int)((unsigned)(KSEG-1)*0x00010001u));
    unsigned long long p = p0;
    for (; p + 256 <= p1; p += 256) {          /* FOUR chains: the reduce is 12 deep */
        const unsigned short *A = S + p*DPAD;
        /* 256 pixels x 2*DPAD bytes, as 128-byte lines.  Getting this height
         * wrong (it is NOT 2*DPAD) covered half the stream and cost 19.0M ->
         * 55.7M cycles -- a silent 2.9x, with the right answer either way. */
        if (p + 512 <= p1) SGM_L2PF(S + (p+256)*DPAD, 128, 128, 4*DPAD);
        HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
        HVX_Vector a2 = Q6_V_vzero(), a3 = Q6_V_vzero();
        for (int i = 0; i < 64; i += AMIN_PPS) {
            const unsigned short *q = A + (size_t)i*DPAD;
            a0 = amin_push(a0, q,                        idx, mk);
            a1 = amin_push(a1, q + (size_t)64*DPAD,      idx, mk);
            a2 = amin_push(a2, q + (size_t)128*DPAD,     idx, mk);
            a3 = amin_push(a3, q + (size_t)192*DPAD,     idx, mk);
        }
        *(HVX_UVector *)(disp + p) =
            Q6_Vb_vpacke_VhVh(AMIN_FIN(a1, mk), AMIN_FIN(a0, mk));
        *(HVX_UVector *)(disp + p + 128) =
            Q6_Vb_vpacke_VhVh(AMIN_FIN(a3, mk), AMIN_FIN(a2, mk));
    }
    for (; p + 128 <= p1; p += 128) {
        const unsigned short *A = S + p*DPAD, *B = S + (p+64)*DPAD;
        HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
        for (int i = 0; i < 64; i += AMIN_PPS) {
            a0 = amin_push(a0, A + (size_t)i*DPAD, idx, mk);
            a1 = amin_push(a1, B + (size_t)i*DPAD, idx, mk);
        }
        *(HVX_UVector *)(disp + p) =
            Q6_Vb_vpacke_VhVh(AMIN_FIN(a1, mk), AMIN_FIN(a0, mk));
    }
    for (; p < p1; p++) {                      /* never taken at our image sizes */
        const unsigned short *Sp = S + p*DPAD;
        unsigned short best = Sp[0]; int bd = 0;
        for (int d = 1; d < SGM_D; d++) if (Sp[d] < best) { best = Sp[d]; bd = d; }
        disp[p] = (unsigned char)bd;
    }
}
/* ============================================================================
 * VPC == 2 : D > 128, so ONE pixel spans two 128-byte vectors (lo = d 0..127,
 * hi = d 128..DPAD-1).  There is no pixel packing left to do, so the only
 * things that change are (a) the min-over-d reduce has to fold the two vectors
 * together first, and (b) the d-1 / d+1 shifts cross the vector boundary --
 * which valign does for free by taking the neighbouring vector as its other
 * operand, so no mask is needed at the seam at all.
 * ==========================================================================*/
#if VPC == 2
static inline void step2v(HVX_Vector Llo, HVX_Vector Lhi,
                          HVX_Vector Clo, HVX_Vector Chi,
                          const pk_t *k, HVX_Vector *olo, HVX_Vector *ohi)
{
    HVX_Vector mn = Q6_Vub_vmin_VubVub(Llo, Lhi);
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(1)));
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(2)));
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(4)));
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(8)));
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(16)));
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(32)));
    mn = Q6_Vub_vmin_VubVub(mn, Q6_V_vdelta_VV(mn, SPL(64)));
    HVX_Vector lolo = Q6_V_valign_VVR(Llo, k->big, 127);  /* d-1, lane0 <- 255      */
    HVX_Vector lohi = Q6_V_valign_VVR(Lhi, Llo,   127);   /* d-1, lane0 <- Llo[127] */
    HVX_Vector hilo = Q6_V_valign_VVR(Lhi, Llo,     1);   /* d+1, lane127 <- Lhi[0] */
    HVX_Vector hihi = Q6_V_valign_VVR(k->big, Lhi,  1);   /* d+1, lane127 <- 255    */
    HVX_Vector p2v = Q6_Vub_vadd_VubVub_sat(mn, k->p2);
    HVX_Vector mlo = Q6_Vub_vmin_VubVub(Llo, Q6_Vub_vadd_VubVub_sat(lolo, k->p1));
    mlo = Q6_Vub_vmin_VubVub(mlo, Q6_Vub_vadd_VubVub_sat(hilo, k->p1));
    mlo = Q6_Vub_vmin_VubVub(mlo, p2v);
    HVX_Vector mhi = Q6_Vub_vmin_VubVub(Lhi, Q6_Vub_vadd_VubVub_sat(lohi, k->p1));
    mhi = Q6_Vub_vmin_VubVub(mhi, Q6_Vub_vadd_VubVub_sat(hihi, k->p1));
    mhi = Q6_Vub_vmin_VubVub(mhi, p2v);
    *olo = Q6_Vub_vadd_VubVub_sat(Clo, Q6_Vb_vsub_VbVb(mlo, mn));
    *ohi = Q6_Vub_vadd_VubVub_sat(Chi, Q6_Vb_vsub_VbVb(mhi, mn));
}

/* pad lanes live only in the HI vector: lane j there is d = 128+j */
static inline void pk_pad2(const pk_t *k, HVX_Vector *plo, HVX_Vector *phi)
{
    unsigned char c[128] __attribute__((aligned(128)));
    for (int i = 0; i < 128; i++) c[i] = (128 + i >= SGM_D) ? 255 : 0;
    (void)k;
    *plo = Q6_V_vzero();
    *phi = *(const HVX_Vector *)c;
}

static inline void cost_group2(const unsigned long long *clr,
                               const unsigned long long *crr, int x,
                               HVX_Vector padlo, HVX_Vector padhi,
                               HVX_Vector *clo, HVX_Vector *chi)
{
    const unsigned char *b = (const unsigned char *)(crr + x - (DPAD-1));
    HVX_Vector c0 = splat8(clr[x]);
    /* entry p sits at address x-(DPAD-1)+p, i.e. d = DPAD-1-p: the first eight
     * r-vectors are the TOP half of the range, the last eight the bottom. */
    HVX_Vector hv = cost_pack8(cost_rvec(b, 0,c0), cost_rvec(b, 1,c0),
                               cost_rvec(b, 2,c0), cost_rvec(b, 3,c0),
                               cost_rvec(b, 4,c0), cost_rvec(b, 5,c0),
                               cost_rvec(b, 6,c0), cost_rvec(b, 7,c0));
    HVX_Vector lv = cost_pack8(cost_rvec(b, 8,c0), cost_rvec(b, 9,c0),
                               cost_rvec(b,10,c0), cost_rvec(b,11,c0),
                               cost_rvec(b,12,c0), cost_rvec(b,13,c0),
                               cost_rvec(b,14,c0), cost_rvec(b,15,c0));
    *chi = Q6_V_vor_VV(Q6_V_vdelta_VV(hv, SPL(127)), padhi);
    *clo = Q6_V_vor_VV(Q6_V_vdelta_VV(lv, SPL(127)), padlo);
}
static inline void cost_group2_edge(const unsigned long long *clr,
                                    const unsigned long long *crr, int x,
                                    HVX_Vector padlo, HVX_Vector padhi,
                                    HVX_Vector dvlo, HVX_Vector dvhi,
                                    HVX_Vector *clo, HVX_Vector *chi)
{
    HVX_Vector a, b;
    cost_group2(clr, crr, x, Q6_V_vzero(), Q6_V_vzero(), &a, &b);
    HVX_Vector t = SPL(x);
    a = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VubVub(dvlo, t), SPL(COST_INVALID), a);
    b = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VubVub(dvhi, t), SPL(COST_INVALID), b);
    *clo = Q6_V_vor_VV(a, padlo);
    *chi = Q6_V_vor_VV(b, padhi);
}

static inline void acc2v(HVX_Vector Llo, HVX_Vector Lhi, HVX_Vector *Sp, int first)
{
    HVX_VectorPair a = widen_pair(Llo), b = widen_pair(Lhi);
    HVX_Vector v0=Q6_V_lo_W(a), v1=Q6_V_hi_W(a), v2=Q6_V_lo_W(b), v3=Q6_V_hi_W(b);
    Sp[0] = first ? v0 : Q6_Vh_vadd_VhVh(Sp[0], v0);
    Sp[1] = first ? v1 : Q6_Vh_vadd_VhVh(Sp[1], v1);
    Sp[2] = first ? v2 : Q6_Vh_vadd_VhVh(Sp[2], v2);
    Sp[3] = first ? v3 : Q6_Vh_vadd_VhVh(Sp[3], v3);
}
static inline void acc2v_add(HVX_Vector Alo, HVX_Vector Ahi,
                             HVX_Vector Blo, HVX_Vector Bhi, HVX_Vector *Sp)
{   /* the fused horizontal sweep: widen and add the two uint8 columns, STORE */
    HVX_VectorPair a = widen_add_pair(Alo, Blo), b = widen_add_pair(Ahi, Bhi);
    Sp[0]=Q6_V_lo_W(a); Sp[1]=Q6_V_hi_W(a); Sp[2]=Q6_V_lo_W(b); Sp[3]=Q6_V_hi_W(b);
}
#endif /* VPC == 2 */

#endif /* SGM_HVX_KERNELS_H */
