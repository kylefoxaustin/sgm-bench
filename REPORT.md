# SGM Stereo — how fast can Arm cores do it, bit-exactly?

**Every number below came from a logged run on real hardware. Nothing is
estimated, scaled, or inherited from a datasheet.** Where a number is derived
it is marked DERIVED; where it comes from someone else's paper it is marked
SOURCED. The rule throughout: a derived or sourced number is never compared
against a measured one without saying so.

---

## The headline

**2,569 MDE/s at 1920×1080 on eight Cortex-A720 cores — 10.6× the fastest
published Arm-CPU SGM, bit-exact.**

Bit-exact means byte-identical to a scalar reference implementation, verified
on every run. The same golden hash (`b1b407b5949f0cc1`) is produced by **seven
independent execution targets**: x86-64, Cortex-A55, Cortex-A720, Cortex-A78C,
two Mali GPUs through OpenCL, and a Hexagon v73 NSP through FastRPC — three
vendors, four instruction sets, three processor classes. A run whose hash does
not match is void and its timing is discarded.

---

## What was measured

1920×1080, disparity range D, 4 aggregation paths (L,R,U,D), 9×7 census
(62-bit descriptor), P1=20, P2=192, integer disparity, no LR check, no
sub-pixel, no median.

**MDE/s** = million disparity estimations per second = `W × H × D / runtime`.
It is the unit the literature uses, because fps hides D: two implementations
can both report "30 fps" while doing 100× different work. Note it normalises
D but **not** census window or path count.

| platform | config | threads | ms | fps | Mpix/s | **MDE/s** | best per-core |
|---|---|---|---|---|---|---|---|
| **8× Cortex-A720** @2.2–2.5 GHz (Radxa O6) | D=64 | 8 | 51.7 | 19.4 | 40.14 | **2,569** | 423 |
| 8× Cortex-A720 | D=128 | 8 | 105.9 | 9.5 | 19.59 | 2,507 | — |
| 1× Cortex-A720 @2.5 GHz | D=64 | 1 | 313.6 | 3.2 | 6.61 | 423 | **423** |
| **6× Cortex-A55** @1.8 GHz (i.MX 95 FRDM) | D=64 | 6 | 341.3 | 2.9 | 6.08 | **389** | 82 |
| **8× Cortex-A78C** (Qualcomm IQ-9075) ‡ | D=64 | **6** | 94.7 | 10.6 | 21.90 | **1,402** | 402 |
| Mali-G720-Immortalis, 10 CU (OpenCL) | D=64 | — | 256 | 3.9 | 8.09 | 518 | — |
| Hexagon v73 NSP (IQ-9075, FastRPC) § | D=64 | 6 | 1223.0 | 0.82 | 1.70 | 108 | — |
| Mali-G310, 1 CU (OpenCL) | D=64 | — | 1846 | 0.54 | 1.12 | 72 | — |
| scalar oracle, 1× A55 (the floor) | D=64 | 1 | 9188 | 0.11 | 0.23 | 14.4 | — |

‡ **Controlled cross-vendor comparison, contributed by the qualcomm session.**
Not an estimate and not a different implementation: they ran *this repository's*
`a55` NEON implementation unmodified, verified the input PGMs on-board against
the published sha256, and reproduced the golden FNV-1a `b1b407b5949f0cc1` at
every thread count. Same D, paths, census window and penalties. MEASURED.

§ **Hexagon row is a different implementation** — an independent HVX/FastRPC
port written by the qualcomm session, not this repository's NEON code. What is
shared is the *specification and the acceptance test*: identical D, paths,
census window, penalties, and the same golden hash `b1b407b5949f0cc1`, held at
1, 2, 4 and 6 DSP threads, with inputs sha256-verified on the board. MEASURED
on their hardware; reproduced here by trust in that hash, not by my own run.

⚠️ Their best is at **six** threads, not eight — 8 threads is *slower* (1,155 vs
1,402 MDE/s). They report the same inversion on an unrelated workload
(llama.cpp), so it is a property of that board, not of this program. **The O6
does not show it**: our A720 scales monotonically 1→2→4→8. Two different
machines, two different answers, both measured.

⭐ **The per-core reading is now controlled rather than indicative: A78C 402 vs
A720 423 MDE/s — a 5% gap that needs no explanation.** Two wide out-of-order Arm
cores from different vendors land in the same place on this algorithm, which is
a stronger statement about SGM than about either core.

### Two wide architectures, two vendors, same loss

The Mali G720 (518 MDE/s) loses to **two** A720 cores. The Hexagon v73 NSP
(108 MDE/s) loses to its own die's A78C cluster by **12.8×** — same board, same
session, same golden. That ratio deliberately uses their *same-session* A78C
re-run (95.51 ms) rather than the 94.7 ms in the table above: comparing two
numbers taken minutes apart on one machine is worth more than comparing across
sessions, and the two agree to 0.6% anyway. Two independent wide-SIMD units, from two vendors,
both beaten by the out-of-order CPU on this algorithm.

The mechanism is the same one that shaped every optimisation in this repo:
**SGM's aggregation is a serial recurrence in x or y.** `L(d)` at one pixel
needs `L(d)` at the previous one, so the only parallelism is across disparities
and across independent scanlines. A wide unit can fill lanes but cannot hide
the dependency, and an OoO core's deep window absorbs exactly this shape.

⚠️ **Honest bounds, and the ratio has already moved once.** The first port
measured 18.7×. Vectorising census — 128 pixels in lanes, accumulated into 8
byte-planes, then interleaved by three rounds of `Q6_W_vshuff_VVR` at byte /
halfword / word granularity — took census from **36.4% of runtime to 6.9%** and
the whole pipeline from 1784.07 to 1222.99 ms, i.e. **1.46× end to end**, hash
unchanged. That is why 18.7× was never quoted as the DSP's floor: it was a floor
of *effort*, and one afternoon moved it to 12.8×.

What remains undone there: aggregation is now 65.7% of runtime and still wastes
**half the vector** (D=64 occupies 64 of 128 byte lanes), with argmin at 14.4%
and cost at 13.0%. The qualcomm session's ceiling for that remaining work is
**~200–250 MDE/s — DERIVED, not measured** — still 5.6–6.9× short of the A78C.

🚨 **A larger caveat, disclosed by them unprompted: all of this is one cDSP of
five co-processors.** The board exposes `/dev/fastrpc-{cdsp,cdsp1,gdsp0,gdsp1,
adsp}`, and the two **gdsps are general-purpose Hexagons that nobody in this
corpus has ever benchmarked**. So 12.8× is a floor of effort on **one of five
engines**, not a statement about the SoC's DSP complex. Work on the other
engines is in flight.

So "the DSP cannot win here" is a *judgement* resting on a measurement, with a
known-incomplete denominator. It is labelled that way rather than promoted into
the headline, and it is the claim most likely to be revised.

⭐ Structural detail worth keeping: the NSP has **six hardware threads but only
four HVX contexts**, read on the board. That is why DSP scaling saturates at
four (3.59× at 4 threads, 3.76× at 6) — a hardware limit, not a code one.

⚠️ **Their trap is our failure shape.** HVX widening is *deinterleaved* — the
low vector holds even-indexed bytes, not bytes 0..63 — so their first kernel was
**3× faster with a wrong hash**: a plausible disparity map, no crash, no zeros,
no warning. It was caught only because this harness gates the hash in the same
run that produces the timing. That design choice has now paid for itself on
hardware I have never touched.

Power: **blank, deliberately.** No board here exposes a usable rail for the
CPU block alone, and rule 6 says leave the cell empty rather than estimate.

---

## 🚨 A defect in this harness, found by someone else

**Until 2026-08-28 the `-t` flag did nothing.** Every implementation takes a
`threads` argument and discards it (`(void)threads`) because the count comes
from OpenMP. So `-t` was parsed, printed in the results table, and written to
the JSON as `"threads":N` — while the run used whatever `OMP_NUM_THREADS` said.
A `-t 1` run emitted a row **labelled single-thread containing an all-core
timing**.

Found by the **qualcomm session**, not by me, on a Cortex-A78C where `-t 1`,
`-t 6` and `-t 8` all returned ~115 ms. They caught it before publishing; had
they not, their single-thread baseline would have been 2.6× too fast — and
their DSP comparison correspondingly flattered.

Controlled A/B on 6× A55 confirming both the defect and the fix (MEASURED,
i.MX 95 FRDM, `OMP_NUM_THREADS` unset, golden `b1b407b5949f0cc1` on all four):

| harness | `-t 1` | `-t 6` | ratio |
|---|---|---|---|
| before | 344.29 ms | 342.65 ms | **1.00× — flag inert** |
| after  | 1588.74 ms | 342.75 ms | 4.63× |

**Note the direction.** The error always shortens the run that was *asked* to
be slow, so it inflates single-thread baselines and thereby flatters whatever
they are compared against. That is the dangerous direction, and it is silent.

⚠️ **And the magnitude is core-dependent: 4.63× on A55, 2.6× on A78C.** The
inert flag understated by however much that particular machine happened to
scale, so **no single correction factor exists** — affected data cannot be
repaired by arithmetic, only by re-running it. (Observation due to the qualcomm
session.)

**The published numbers above are unaffected**, and the reason is not "our
wrapper set `OMP_NUM_THREADS`" (it did — `scripts/pin.sh:57` — but that is luck,
not design). It is that **under this bug every thread count collapses to the
same timing**, and the tables do not: A720 spreads 6.07× from 1→8 threads, A55
4.65× from 1→6. Real scaling is proof the flag was working on those runs.
Independent confirmation: the A55 per-core figure of 82 MDE/s published before
the bug was known matches the 83.5 MDE/s measured today while testing the fix.

Fixed in `common/harness.c`: `-t` now calls `omp_set_num_threads()`, and the
count that reaches the table and the JSON is **read back** from the runtime
(`omp_get_max_threads()`) rather than echoed from the command line, with the
requested value kept alongside as `threads_requested`. A future divergence
surfaces as a warning instead of a plausible number.

---

## Against published work

SOURCED — ReS2tAC, *Sensors* 21(11):3938, 2021, Tables 6 and 7.

| implementation | D | MDE/s | hardware | class |
|---|---|---|---|---|
| Zhao et al., CT5×5-SGM | 128 | 9,590 | FPGA | fixed-function |
| Rahnama et al., CT5×5-MGM | 128 | 4,247 | FPGA | fixed-function |
| **this work** | **128** | **2,507** | **8× A720** | **CPU** |
| Hernandez-Juarez, CT9×7-SGM | 128 | 747 | Jetson TX1 | GPU |
| ReS2tAC-CUDA, CT9×7-SGM | 128 | 645 | Xavier AGX | GPU |
| **this work — Mali G720** | 64 | 518 | 10 CU | GPU |
| ReS2tAC-NEON, **4-path** CT5×5 | 128 | 242 | 8× Carmel | CPU |
| ReS2tAC-NEON, 8-path CT5×5 | 128 | 166 | 8× Carmel | CPU |

**10.4× the published Arm-CPU record at matched D and matched path count**, and
**3.4× the best embedded-GPU SGM in that table — from a CPU.**

⚠️ **Caveat, and it cuts both ways.** The controlled axes (D, path count) are
matched. The cost function is not: ours is a 9×7 / 62-bit census where theirs is
5×5 / 24-bit, so **we do ~2.6× more Hamming work per pixel** — which makes the
throughput comparison conservative. But image content and harness also differ,
so this is *indicative*, not a controlled head-to-head.

---

---

## What worked, and what didn't

Nine optimisations were tried. **Four survived measurement; five were rejected.**
That ratio is the honest shape of the work.

### Kept

| change | effect | why it worked |
|---|---|---|
| Vectorised argmin | 1.54× | replaced 64 scalar iterations/pixel (132M/frame) |
| Blocked column sweep | 2.4× | fixed a stride-15 KB access pattern |
| Block-width tuning | ~1.25× | A55 optimum 192, A720 optimum **256** — *not* the same |
| Non-temporal stores + 64B alignment | 2.8% | planes are write-once/read-once; malloc gave page+16 |

### Rejected — each with its mechanism

| change | effect | why it failed |
|---|---|---|
| Software prefetch | **+0.7% worse** | prefetched a stream the HW prefetcher already had |
| Interleaved `[D\|U]` plane | **+13% worse** | strided writes break full-line bursts; L3 refills +17% |
| Scanline interleaving (K=2,3,4) | **0 to −3%** | column blocking already filled the OoO window (IPC 2.63) |
| Contiguous row bands | *rejected in design* | band *b+1* waits on band *b*'s last row = serialisation |
| Per-row wavefront | **2.9× worse** | dependency granule = parallel unit ⇒ pipeline depth 1 |

### Three rules the failures produced

1. **A wavefront only buys parallelism when the dependency granule is smaller
   than the parallel unit.** Row-to-row dependency with row-sized work units is
   just serial execution with extra atomics.
2. **Strided writes cost more than stream count saves.** Losing sequential
   full-cache-line bursts outweighed removing an entire read stream.
3. **The 133 MB materialised plane is not waste — it is the price of
   parallelising a vertical recurrence.** The U path has no column dependency,
   so it parallelises perfectly over columns; every attempt to trade the plane
   away cost more than it saved.

---

## Corrections to the specification

Found by measurement, listed so the spec can be fixed:

- **A55 L2 is 64 KB/core**, not large enough for the 123 KB row buffers the spec
  assumes will fit. (L3 is 512 KB shared by six cores.)
- **The O6 has four A720 frequency pairs** (2.5/2.4/2.3/2.2 GHz), not "two
  clusters". No four A720s share a clock, so a clean homogeneous 4-thread
  per-core measurement is impossible on that part.
- **gcc 12.2 cannot target `cortex-a720`.** All A720 results use
  `-mcpu=cortex-a710` (same Armv9 family, same 4×128-bit SIMD). Documented
  deviation; a gcc-13+ build is untested.
- **"No compute driver" was not the answer for either GPU.** Both ship working
  OpenCL 3.0. The G310 merely lacks `clinfo`, which makes it *look* driverless.
- **9×7 census is 62 bits, not 63** (centre excluded). Code was right; two
  comments and the spec table were wrong.

---

## Open items

- **Phase 5 (libSGM on RTX 5090)** not run — the GPU has been under another
  session's campaign throughout.
- **Power** unmeasured on every platform; the perf/W column cannot be filled
  from these boards.
- **8-path** numbers not taken; all results are 4-path. The literature's 8→4
  reduction is ×1.45 for +0.2% error (SOURCED), so an 8-path figure would be
  roughly 0.69× these.
- **A gcc-13+ A720 rebuild** could move the headline number in either direction.
- **Real stereo pair** not benchmarked — all results are on the deterministic
  synthetic pair. The spec warns against tuning on synthetic only; census
  bit-exactness was verified on both synthetic and real image statistics, but
  the timings are synthetic-only.
