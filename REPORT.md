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
on every run. The same golden hash (`b1b407b5949f0cc1`) is produced by **five
independent execution targets**: x86-64, Cortex-A55, Cortex-A720, and two Mali
GPUs through OpenCL. A run whose hash does not match is void and its timing is
discarded.

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
| **8× Cortex-A78C** (Qualcomm IQ-9075) ‡ | D=64 | **6** | 94.7 | 10.6 | 3.79 | **1,402** | 402 |
| Mali-G720-Immortalis, 10 CU (OpenCL) | D=64 | — | 256 | 3.9 | 8.09 | 518 | — |
| Mali-G310, 1 CU (OpenCL) | D=64 | — | 1846 | 0.5 | 1.12 | 72 | — |
| scalar oracle, 1× A55 (the floor) | D=64 | 1 | 9188 | 0.11 | 0.23 | 14 | — |

‡ **Controlled cross-vendor comparison, contributed by the qualcomm session.**
Not an estimate and not a different implementation: they ran *this repository's*
`a55` NEON implementation unmodified, verified the input PGMs on-board against
the published sha256, and reproduced the golden FNV-1a `b1b407b5949f0cc1` at
every thread count. Same D, paths, census window and penalties. MEASURED.

⚠️ Their best is at **six** threads, not eight — 8 threads is *slower* (1,155 vs
1,402 MDE/s). They report the same inversion on an unrelated workload
(llama.cpp), so it is a property of that board, not of this program. **The O6
does not show it**: our A720 scales monotonically 1→2→4→8. Two different
machines, two different answers, both measured.

⭐ **The per-core reading is now controlled rather than indicative: A78C 402 vs
A720 423 MDE/s — a 5% gap that needs no explanation.** Two wide out-of-order Arm
cores from different vendors land in the same place on this algorithm, which is
a stronger statement about SGM than about either core.

Power: **blank, deliberately.** No board here exposes a usable rail for the
CPU block alone, and rule 6 says leave the cell empty rather than estimate.

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
