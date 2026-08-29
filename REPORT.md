# SGM Stereo — how fast can Arm cores do it, bit-exactly?

**Every number below came from a logged run on real hardware. Nothing is
estimated, scaled, or inherited from a datasheet.** Where a number is derived
it is marked DERIVED; where it comes from someone else's paper it is marked
SOURCED. The rule throughout: a derived or sourced number is never compared
against a measured one without saying so.

---

## The headline

**2,569 MDE/s at 1920×1080 on eight Cortex-A720 cores — 10.6× the fastest
published Arm-CPU SGM, bit-exact.** That remains the headline because it is the
*CPU* result and the literature comparison is against CPUs; a Jetson Thor doing
3,750 MDE/s through CUDA now sits above it in the table, and is a different
claim about different silicon.

Bit-exact means byte-identical to a scalar reference implementation, verified
on every run. The same golden hash (`b1b407b5949f0cc1`) is produced by **nine
independent execution targets**: x86-64, Cortex-A55, Cortex-A720, Cortex-A78C,
two Mali GPUs through OpenCL, a Hexagon v73 NSP through FastRPC, and two NVIDIA
GPUs through CUDA — four vendors, four instruction sets, three processor
classes. A run whose hash does
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

⚠️ **And it treats D as free-floating when the optics fix it.** Disparity is
`d = f·B/Z` with `f = (W/2)/tan(HFOV/2)`, so the D you *need* scales with image
width and inversely with field of view. Since work is `W·H·D` and `D ∝ W`, **SGM
work goes as W²·H — cubic in linear resolution, not quadratic.** Dropping 1080p
to 720p is 2.25× fewer pixels but ~3.4× less work. Verified independently here;
the geometry is due to the qualcomm session.

Three consequences a throughput number hides:

- **A wider field of view is cheaper.** 90° → 120° cuts focal length 1.73×, cuts
  required D by 1.73×, and *sees more of the scene*. FOV is a performance knob
  and nobody treats it as one.
- **This benchmark's configuration is easier than a robot's.** At a 12 cm
  baseline and 90°, D=64 implies a **1.80 m minimum range** — useless for
  manipulation. A deployable 720p/120°/D=128 configuration reaches 0.35 m.
- **Bigger D is cheaper per disparity and more expensive per frame.** Our own
  A720 data says so: doubling D=64→128 costs 2.05× wall clock for 2.00× the
  disparities, a −2.4% change in MDE/s. On the NSP, where D=64 leaves a 128-lane
  vector half empty, larger D is outright *faster* per disparity (773 / 987 /
  1121 / 1198 MDE/s at D = 32 / 64 / 128 / 256 — MEASURED by the qualcomm
  session on their hardware).

So MDE/s is the right unit for comparing implementations at a fixed
configuration, and the wrong unit for choosing a configuration. The binding
constraint on a real system turns out to be the **camera baseline**, not the
kernel.

### Two configuration traps

MEASURED by the qualcomm session on the Hexagon NSP; their arithmetic
re-checked here for internal consistency before being reproduced.

🚨 **D pads to the next power of two, so D=96 and D=240 should never be built.**
D=240 measured 442.62 ms against D=256's 443.04 ms — 0.09% apart, identical
memory, **16 disparities available for free**. A configuration that silently
charges full price for a partial range is the same shape as everything else this
project found today: correct output, no complaint, quietly worse. Their build
`#error`s above D=256 rather than producing a silently wrong map, which is the
right instinct — a compile-time refusal beats a plausible answer.

⚠️ **Efficiency falls as resolution falls.** 1080p 962 → 720p 756 → 540p 704 →
VGA 631 MDE/s. Dropping resolution buys **latency, not throughput per
disparity**, because fixed overheads amortise over fewer pixels: 1080p→720p is
1.77× wall-clock against a 2.25× pixel reduction. Combined with the geometry
above, a lower-resolution system is doing less total work but each disparity
costs more.

And one engine that could not be tested at all: the **EVA/CVP** vision
accelerator is present in silicon with 7 MB reserved at boot, but the Yocto
image ships **no firmware and no driver**, so it cannot be started. Recorded as
untestable rather than absent — it is the one engine where a fixed-function
design would be expected to win, by keeping the cost volume on-chip and never
paying the DDR round trip that dominates every implementation measured here.

| platform | config | threads | ms | fps | Mpix/s | **MDE/s** | best per-core |
|---|---|---|---|---|---|---|---|
| **NVIDIA Thor** (Blackwell, 20 SM, CUDA, tuned) ¶ | D=64 | — | 13.2 | 75.98 | 157.55 | **10,084** | — |
| NVIDIA Thor — D=128, the grid's peak ¶ | D=128 | — | 21.6 | 46.25 | 95.91 | **12,276** | — |
| NVIDIA Thor — naive OpenCL transliteration ¶ | D=64 | — | 35.4 | 28.26 | 58.59 | 3,750 | — |
| **8× Cortex-A720** @2.2–2.5 GHz (Radxa O6) | D=64 | 8 | 51.7 | 19.4 | 40.14 | **2,569** | 423 |
| 8× Cortex-A720 | D=128 | 8 | 105.9 | 9.5 | 19.59 | 2,507 | — |
| 1× Cortex-A720 @2.5 GHz | D=64 | 1 | 313.6 | 3.2 | 6.61 | 423 | **423** |
| **6× Cortex-A55** @1.8 GHz (i.MX 95 FRDM) | D=64 | 6 | 341.3 | 2.9 | 6.08 | **389** | 82 |
| **NVIDIA Jetson AGX Orin** (Ampere, 16 SM, CUDA, tuned) ¶ | D=64 | — | 23.3 | 42.95 | 89.06 | **5,700** | — |
| **8× Cortex-A78C** (Qualcomm IQ-9075) ‡ | D=64 | **6** | 94.7 | 10.6 | 21.90 | **1,402** | 402 |
| Mali-G720-Immortalis, 10 CU (OpenCL) | D=64 | — | 256 | 3.9 | 8.09 | 518 | — |
| **Hexagon v73 NSP** (IQ-9075, FastRPC) § | D=64 | **4** | 138.6 | 7.21 | 14.96 | **957** | — |
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

¶ **CUDA rows are a deliberate one-for-one port of `mali_cl/sgm.cl`**, not a
rewrite: that kernel was already verified bit-exact on two Mali parts, so
porting it mechanically carries the semantics across rather than re-deriving
them and re-meeting the same tie-break and edge-replication traps. It was
bit-exact on the first run on both parts. **A block is 16 threads — half a
warp — which is deliberately poor CUDA occupancy**, so these are floors for
these GPUs in exactly the sense the Mali number is a floor for the G720.
MEASURED, 20 timed frames.

⚠️ Their best is at **six** threads, not eight — 8 threads is *slower* (1,155 vs
1,402 MDE/s). They report the same inversion on an unrelated workload
(llama.cpp), so it is a property of that board, not of this program. **The O6
does not show it**: our A720 scales monotonically 1→2→4→8. Two different
machines, two different answers, both measured.

⭐ **The per-core reading is now controlled rather than indicative: A78C 402 vs
A720 423 MDE/s — a 5% gap that needs no explanation.** Two wide out-of-order Arm
cores from different vendors land in the same place on this algorithm, which is
a stronger statement about SGM than about either core.

### Optimising the CUDA kernel: 2.7× to a measured memory wall

`cuda/sgm_cuda.cu` was a transliteration of the OpenCL kernel and inherited its
shape: 16-thread blocks (half a warp), three `__syncthreads()` **per pixel**
inside the recurrence, `L` and the min-reduction in shared memory, and 1,080
blocks of 16 threads to fill a 20-SM GPU. `cuda/sgm_cuda_opt.cu` keeps the
algorithm and the hash and changes all of it.

| step | Thor | | mechanism |
|---|---|---|---|
| naive port | 35.39 ms | — | baseline |
| warp-per-scanline, `L` in registers, `__shfl` reduce, **zero barriers** | 16.58 ms | 2.13× | the recurrence stops touching shared memory |
| first path **stores** instead of accumulating; `ushort2` S access | 14.61 ms | 2.42× | deletes a 265 MB memset and a 265 MB read |
| pair the paths through a uint8 scratch: S touched 4× not 7× | **13.16 ms** | **2.69×** | 1.99 GB → 1.46 GB of traffic |

Orin gets **3.10×** from the same changes (72.27 → 23.28 ms). Every step
bit-exact; a step that broke the hash would not be in this table.

⭐ **It is now memory-bound, and that is measured rather than asserted.** A
copy-bandwidth probe on the same part returns **247 GB/s**. The aggregate phase
moves 1.46 GB in 10.04 ms = **145 GB/s**. Before the pairing it moved 1.99 GB in
11.37 ms = 175 GB/s — so cutting traffic 27% bought only 12% of time, because
the uint8 scratch is a *less efficient* access pattern than the S it replaced.
Net win, roughly half of what the traffic model predicted. **Arithmetic
optimisation is finished here; only traffic reduction can move it further.**

⚠️ Occupancy is not a lever: warps-per-block of 2, 4 and 8 land within 1%
(14.71 / 14.59 / 14.71 ms) and 32 is clearly worse. Swept rather than guessed.

### The parameter grid, and one flaw it exposed in the sweep itself

MEASURED on Thor, every cell a separate build (D is compile-time) and a separate
golden (changing D changes the correct answer):

| resolution | D=32 | D=64 | D=128 |
|---|---|---|---|
| 1920×1080 | 6,233 | 10,084 | **12,276** |
| 1280×720 | 5,617 | 9,504 | 12,057 |
| 960×540 | 4,079 | 8,791 | 10,835 |
| 640×480 | 4,072 | 5,108 | 10,956 |

⭐ **A larger disparity range is cheaper per disparity** — 6,233 → 10,084 →
12,276 MDE/s as D goes 32 → 64 → 128 at 1080p. And **efficiency falls as
resolution falls**: at D=64, 10,084 → 9,504 → 8,791 → 5,108. Both reproduce what
the qualcomm session measured on a Hexagon NSP, on entirely different silicon
with a different implementation. Lower resolution buys **latency, not throughput
per disparity**.

🚨 **The first run of this grid produced identical hashes at D=64 and D=128, and
that is exactly the failure qualcomm named: a check that cannot discriminate.**
It is also what a kernel searching only `d < 64` would produce. Rather than
accept twelve `GOLDEN OK` lines, the discriminating test was to read the maximum
disparity actually present in the D=128 output: **44**. The cause was mine —
`gen_synthetic.c` scales its scene disparities with `SGM_D`, and the sweep built
the generator *without* `-DSGM_D`, so every cell got a D=64 scene that gave D=128
nothing to find. Fixed; all twelve hashes are now distinct and the D=128 cells
genuinely exercise the range. The timings were never wrong — SGM's work is
data-independent — but the verification was worth nothing until the scene matched
the range under test.

### Why D gets cheaper per disparity — three architectures, three mechanisms

The same observation shows up on NEON, HVX and CUDA and does **not** have the
same cause on any two of them. The qualcomm session supplied the first two; the
third is ours and required falsifying a wrong guess first.

| | lanes | D=64→128 | mechanism |
|---|---|---|---|
| **NEON** (A720) | 16 bytes | **−2.4%, flat** | a vector is already FULL at D=64; extra disparities cost proportional time |
| **HVX** (Hexagon) | 128 bytes | **+13.6%** | at D=64 the vector is HALF EMPTY; raising D fills waste, so it is nearly free |
| **CUDA** (Thor) | warp always covers D | **+22%** | neither — see below |

On CUDA the lanes are never idle: a warp covers exactly D at DPT=D/32
disparities per lane. So the lane-filling model predicts *flat*, like NEON, and
the measurement says otherwise.

🔴 **The first explanation was wrong and was tested rather than argued.** The
guess was that only DPT==2 took the vectorised `ushort2` S path while D=32 and
D=128 fell back to a scalar loop — which was *true of the code* and would have
handicapped exactly the two cells at the ends of the sweep. Generalising the
vector path to DPT 1/2/4 changed nothing: 10.59 / 13.19 / 21.72 ms, all inside
noise. nvcc was already emitting efficient accesses for the contiguous loop. The
generalisation is kept because it is more honest code, not because it helped.

⭐ **The actual mechanism is a fixed per-pixel cost amortised over DPT.** Every
recurrence step does a 5-step `__shfl_xor` min-reduction plus 2 neighbour
shuffles — **7 shuffles per pixel per path, independent of D** — serving DPT
disparities per lane. Measured:

| D | aggregate ms **per disparity** | census (D-independent) |
|---|---|---|
| 32 | 0.2375 | 1.08 ms = 10% of total |
| 64 | 0.1604 | 1.08 ms = 8% |
| 128 | 0.1424 | 1.14 ms = 5% |

Census is flat in absolute terms and simply amortises. But *aggregate itself*
gets cheaper per disparity, which census cannot explain — that is the warp
reduction being divided over 1, then 2, then 4 disparities per lane.

So it is the same *family* as the Hexagon result — a fixed cost spread over more
useful work — while being a different quantity: idle lanes there, an
irreducible cross-lane reduction here. And the effect is decelerating (1.48×
then 1.13×), as it must be once the shuffle is no longer the dominant term.

### Range and field of view, DERIVED from the grid

FOV is not a measured axis and must not be one: it does not change the kernel, it
changes the D a given near range requires. At a 12 cm baseline:

| configuration | min range | fps |
|---|---|---|
| 1080p, 90°, D=32 | 3.60 m | 94 |
| 1080p, 90°, D=64 | 1.80 m | 76 |
| 1080p, 90°, D=128 | 0.90 m | 46 |
| 1080p, 120°, D=64 | 1.04 m | 76 |

**A wider field of view is cheaper.** 90° → 120° cuts focal length 1.73×, cuts
the D needed for the same near range by 1.73×, and sees more of the scene at the
same cost. Nobody treats FOV as a performance knob; on this workload it is one.
And the binding constraint on a real system is the **camera baseline**, not the
kernel — every row above moves if B changes.

### The NVIDIA GPUs, and what they do and do not settle

Two CUDA targets were added by porting `mali_cl/sgm.cl` one-for-one. Both were
bit-exact on the first run, which is the payoff for porting a verified kernel
instead of writing a new one.

| | ms | MDE/s | vs |
|---|---|---|---|
| Jetson Thor (Blackwell, 20 SM) | 35.4 | **3,750** | 1.46× the 8× A720 |
| Jetson AGX Orin (Ampere, 16 SM) | 72.3 | **1,836** | 3.54× the Mali G720 |

⚠️ **Neither number says a GPU beats a CPU at SGM.** Both are floors, and
knowingly so: a block is 16 threads, half a warp, chosen to mirror the OpenCL
kernel's 16-wide work-group rather than to suit CUDA. Nobody has tuned occupancy,
tiling, or memory layout for these parts. What they establish is that **the
golden survives a fourth vendor and a third programming model** — and that the
recurrence which punished the Mali does not punish a GPU with enough parallelism
across scanlines to hide it.

⭐ **The cost gate is diagnostic even when it does not fire.** Its own ns/op
output shows Orin's argmin at 0.0535 against census at 0.0137 — **3.9× less
efficient than the cheapest phase in the same kernel**. That is the serial
64-iteration argmin loop, one thread per pixel, unvectorised and unreduced. The
gate passes it because it is calibrated against itself, which is exactly the
documented limitation: it catches regression from the calibration point, not an
implementation that was already slow when calibrated. The *numbers* still name
the phase to fix.

### 🔴 The wide-unit finding was wrong, and it took three revisions to find out

**This section previously concluded that both wide-SIMD targets lose to the
out-of-order CPUs. That conclusion is withdrawn for the DSP.**

The Hexagon NSP measured **18.7× slower than the A78C cluster, then 12.8×, then
1.45× — a 13× move in one day**, all bit-exact against the same golden. At its
current best it does **957 MDE/s at 4 threads**, and it **beats a single A78C
core by 2.40×**. Nothing here was retracted along the way, because the ratio was
labelled a floor of *effort* rather than the DSP's ceiling every time it was
printed. That labelling is the only reason this reads as progress instead of as
three corrections.

⚠️ **A DERIVED ceiling was wrong by ~4×, which is the cautionary part.** The
qualcomm session estimated ~200–250 MDE/s as a generous bound on the remaining
work, tagged DERIVED, and the judgement "the DSP cannot win here" rested on it.
The answer was 957. The estimate failed by scaling the phases that were visible
and assuming the rest was incremental: the largest single win turned out to be a
**265 MB memset that had never been profiled**, and the second was using **4
threads instead of 6** — a correction to `num_hvx128_contexts`, a parameter that
had *already been measured and then not acted on*. **A ceiling derived from the
costs you can already see is not a ceiling; it is a restatement of your current
profile.**

### What the mechanism actually is

The original claim — SGM's aggregation is a serial recurrence, so wide units can
fill lanes but cannot hide the dependency — was right about the *shape* and wrong
about the *conclusion*. The correction:

**The latency is hideable, by interleaving independent dependency chains, at a
cost in registers.** A DSP-side bandwidth probe found 28.17 GB/s available at 4
threads while aggregation consumed 6.8 — it was **latency-bound, not
bandwidth-bound**. Redirecting to op-count, prefetch and independent chains was
worth **6.3× on argmin** and **2.5× on the vertical paths**.

⚠️ **Which means our own Mali number is probably also a floor, and by the same
mechanism.** `mali_cl/sgm.cl` carries **one** `uchar4` dependency chain per
work-item across the entire sweep, with three barrier sites per recurrence step.
It never interleaves chains. The 518 MDE/s figure stands as MEASURED, but it
should not be read as what the G720 can do — only as what this kernel does. That
is a limitation of our code, stated here rather than left for someone to find.

🚨 **And that floor is UNBOUNDED, which makes it a weaker statement than it
looks.** The mechanism predicts our Mali number is low; it does not predict by
how much. Nobody has done for the G720 what was done for the DSP, so the honest
position is that the G720's capability on this workload is *unmeasured* — not
that it is competitive. "This kernel leaves performance on the table" and "the
GPU would win if written properly" are different claims and only the first is
supported.

Interleaving is not free and does not always win: on the *horizontal* paths 4
chains were **worse** (22.1M → 40.2M cycles), because horizontal chains must
pair rows, so 4 chains means 16 open streams. Vertical chains walk one contiguous
region and win. Prefetch distance 64 was optimal; 128/256/512 gave 206/223/255 ms.

### 🚨 Only a predicted cost catches slow-but-right

**This benchmark shipped one of the two gates every kernel needs, for a full
day, while believing it had shipped the important one.**

A golden hash is a statement about *outputs*. It cannot constrain how they were
produced, so it is structurally blind to a kernel that is correct and slow. The
second gate is a *predicted cost*, and the campaign had none.

⚠️ **And my own reasoning here was half wrong, in a way worth recording.** When
the census speedup was disputed I chose to publish the **shares** (36.4% → 6.9%)
over an inferred speedup, arguing shares "need no denominator and cannot rot".
The first half was right — the inference was the weaker method. The second half
was wrong: **a share is a ratio against a total that also moves**, so a phase can
read 6.9% and look finished while hiding a 5.8×. Shares were the right way to
report the *change* and a bad instrument for judging whether a phase is *done*.
Exactly the failure mode of the redundant `Mpix/s` column — a derived number
nothing re-derives.

⚠️ **A third way a check can be green and worthless, which we had not named:
it can be unable to discriminate.** The qualcomm session gated a new D=128 path
on "must match D=64's hash" — a condition **also satisfied by a kernel that only
ever searched d < 64**. The check ran, and passed, and could not have failed for
the bug it existed to catch. That completes the set:

| failure mode | example from this campaign |
|---|---|
| a check that **cannot fail** | the inert `-t` flag — 4.63× understatement on A55 |
| a check **nobody runs** | absolute per-phase ms, in the JSON since commit 1 |
| a check that **cannot discriminate** | "D=128 must match D=64's hash" |
| **slow-but-right** — beyond correctness gating entirely | two half-scalar phases |

All four read as green from outside. The last one is the only class a golden
file can never address even in principle.

Three detectors for it, in increasing cost (due to the qualcomm session):

1. **Absolute per-phase cost, never shares.** The harness has written absolute
   `census_ms` / `cost_ms` / `aggregate_ms` / `argmin_ms` into the JSON from the
   start. The data existed for the whole campaign; nothing gated on it.
2. **An op-count roofline per phase.** Census is 62 compares over 128 lanes —
   its cost is computable within ~2× from first principles. Gate on *that*
   rather than on "faster than last time"; anything far off its own arithmetic
   is hiding a scalar path, a spill, or a tail.
3. **Vector-instruction fraction.** Disassemble the hot function and count
   vector ops. A kernel 6.7% scalar *by pixel* is nowhere near 6.7% scalar by
   instruction, because the scalar path is a 62-iteration inner loop.

🚨 **Pick the threshold by planting a defect, not by taste.** A roofline tuned
2× too generous, or keyed to a phase that no longer exists after a fusion, passes
everything forever and looks exactly like a clean codebase. Note that this
harness already reports `cost_ms = -1` when cost is fused into aggregation and
`argmin_ms = -1` when argmin is fused — so a phase-keyed gate would silently skip
precisely the phases most likely to be hiding something. **It must fail closed on
a missing phase, not skip it.** A green check nobody has ever seen go red is not
evidence.

### The case that motivated it: a bit-exact half-scalar kernel

The single most useful thing to come out of the DSP work, and it is a limit on
this repo's whole acceptance model:

Their census kernel stepped 128 columns from x=4 and **stopped at 1796**, handing
the last 124 columns of every row to a scalar 62-neighbour fallback — **6.7% of
pixels consuming ~50% of the phase**. One overlapping tail block took it from
92.0M to 15.8M cycles.

**The hash check caught nothing, because the output was correct.** Correctness
gating proves a kernel computes the right answer. It says nothing about whether
it computes it the fast way. A vectorised kernel with a scalar tail is bit-exact
and slow, and no golden file will ever tell you. Profile the phases; do not infer
from the fact that it passes.

### The GDSPs: a negative result on an engine nobody had measured

The board exposes five co-processors (`cdsp`, `cdsp1`, `gdsp0`, `gdsp1`, `adsp`).
The two GDSPs are general-purpose Hexagons and **they have no HVX at all** —
`HVX_SUPPORT_128B=0`, `VTCM=0`, `ARCH_VER 0x8273`, and any HVX path returns
`0x8000040d`, against the cDSP's 4 HVX contexts and 8 MB VTCM. Scalar SGM runs
on gdsp0 bit-exact (320×240, hash `3e2d0be6b4a6b6dd`, 840.70 ms), and on
identical scalar code a GDSP is **1.24× slower** than a cDSP — 1.086× from clock
(1340 vs 1455 MHz) and 1.143× from cycles. Six-thread scalar v73 cores: useless
for SGM, real cores for scalar offload.

Dual-NSP gives **1.79× throughput** (6.17 → 11.04 fps) but single-frame latency
**11–15% worse** (137 → 153 ms) from DDR contention. A bit-exact single-frame
split is blocked not by the algorithm but by FastRPC cache maintenance on a
411 MB in/out buffer at each barrier. Recorded as UNBUILT rather than estimated.

⭐ Structural detail that turned out to matter: the NSP has **six hardware
threads but only four HVX contexts**. The best configuration is **4 threads, not
6** — and that parameter had been measured before it was acted on.

⚠️ **The failure shape this project keeps meeting.** HVX widening is
*deinterleaved* — the low vector holds even-indexed bytes, not bytes 0..63 — so
an early kernel was **3× faster with a wrong hash**: a plausible disparity map,
no crash, no zeros. Later, a second build in a shared tree produced 246 ms
against a known-good 1223 ms, again with a wrong hash, and again was caught only
because the harness gates the hash in the same run as the timing. Two saves on
hardware this repo's author has never touched.

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
