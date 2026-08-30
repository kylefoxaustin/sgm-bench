# SGM Stereo — how fast can Arm cores do it, bit-exactly?

**Every number below came from a logged run on real hardware. Nothing is
estimated, scaled, or inherited from a datasheet.** Where a number is derived
it is marked DERIVED; where it comes from someone else's paper it is marked
SOURCED. The rule throughout: a derived or sourced number is never compared
against a measured one without saying so.

---

## The headline

**2,507 MDE/s at 1920×1080 D=128 on eight Cortex-A720 cores — 10.4× the
fastest published Arm-CPU SGM, bit-exact, at MATCHED D and matched path
count.** (Our D=64 figure is 2,569, but the published comparator is a D=128
number, so the matched pair is the only honest ratio.) That remains the headline because it is the
*CPU* result and the literature comparison is against CPUs; an RTX 5090 doing
37,383 MDE/s through CUDA now sits above it in the table, and is a different
claim about different silicon.

Bit-exact means byte-identical to a scalar reference implementation, verified
on every run. The same golden hash (`46470bd7a464469d`) is produced by **nine
execution targets**: x86-64, Cortex-A55, Cortex-A720, Cortex-A78C, two Mali GPUs
through OpenCL, a Hexagon v73 NSP through FastRPC, and three NVIDIA GPUs through
CUDA — four vendors, four instruction sets, three processor classes.

⚠️ **"Independent" would overstate it.** These ten targets run roughly four
codebases: the scalar reference, the NEON implementation (shared unmodified by
A55, A720 and A78C), the OpenCL kernel and its mechanical CUDA port, and the
independently-written HVX kernel. The hash pins "matches this reference", not
"is SGM" — a shared misreading of the specification would pass on all of them.
Genuine independence exists for the HVX port and partially for OpenCL. A run whose hash does
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
| **NVIDIA Thor** (Blackwell, 20 SM, CUDA, tuned) ¶ | D=64 | — | 13.2 | 75.35 | 156.24 | **10,062** | — |
| NVIDIA Thor — D=128, the grid's peak ¶ | D=128 | — | 21.65 | 46.19 | 95.78 | **12,260** | — |
| NVIDIA Thor — naive OpenCL transliteration ¶ | D=64 | — | 35.4 | 28.26 | 58.59 | 3,750 | — |
| **8× Cortex-A720** @2.2–2.5 GHz (Radxa O6) | D=64 | 8 | 51.7 | 19.4 | 40.14 | **2,569** | 423 |
| 8× Cortex-A720 | D=128 | 8 | 105.9 | 9.5 | 19.59 | 2,507 | — |
| 1× Cortex-A720 @2.5 GHz | D=64 | 1 | 313.6 | 3.2 | 6.61 | 423 | **423** |
| **6× Cortex-A55** @1.8 GHz (i.MX 95 FRDM) | D=64 | 6 | 341.3 | 2.9 | 6.08 | **389** | 82 |
| **NVIDIA Jetson AGX Orin** (Ampere, 16 SM, CUDA, tuned) ¶ | D=64 | — | 23.3 | 42.95 | 89.06 | **5,700** | — |
| **8× Cortex-A78C** (Qualcomm IQ-9075) ‡ | D=64 | **6** | 98.0 | 10.2 | 21.16 | **1,354** | 395 |
| Mali-G720-Immortalis, 10 CU (OpenCL) | D=64 | — | 256 | 3.9 | 8.09 | 518 | — |
| **Hexagon v73 NSP** (IQ-9075, FastRPC) § | D=64 | **4** | 136.9 | 7.30 | 15.14 | **969** | — |
| Mali-G310, 1 CU (OpenCL) | D=64 | — | 1846 | 0.54 | 1.12 | 72 | — |
| scalar oracle, 1× A55 (the floor) | D=64 | 1 | 9188 | 0.11 | 0.23 | 14.4 | — |

‡ **Controlled cross-vendor comparison, contributed by the qualcomm session.**
Not an estimate and not a different implementation: they ran *this repository's*
`a55` NEON implementation unmodified, verified the input PGMs on-board against
the published sha256, and reproduced the golden FNV-1a `46470bd7a464469d` at
every thread count. Same D, paths, census window and penalties. MEASURED.

§ 🚨 **Every single-invocation Hexagon number carries roughly ±10%, and that is
a property of the board rather than of the measurement.** The qualcomm session
found the multi-modality is *between* invocations, not within them, so a larger
`-n` inside one invocation cannot average it out — it needs interleaved repeats
across invocations. They demonstrated it by interleaving the old and corrected
scenes three times in one session: both spanned the same 135–150 ms, minima
agreeing within 0.5%, which is how they established that a 9.8% apparent
"regression" on the corrected scene was board state and not content. Quote the
Hexagon row accordingly.

§ ⭐ **Re-sourced 2026-08-29 onto a golden that discriminates the full range.**
The originally published 137.91 ms / 962 MDE/s was measured on this repo's
`data/synthetic`, whose golden could not see `d >= 45` (above). The qualcomm
session's own D-parametric work had independently built a discriminating scene
(`data/wide1080`, true disparities to 154) on which **all 64 disparity values win
somewhere**, and their kernel reproduces that golden bit-exactly at
**136.84 ms / 969.8 MDE/s**, hash `33b80d5d35e0fb9b`. The number barely moved;
the *evidence* went from partial to complete, which is the point. Their D=128
figure, 238.89 ms / 1111 MDE/s (`fa2238cc8a87af3d`), was always on that scene.

§ **Hexagon row is a different implementation** — an independent HVX/FastRPC
port written by the qualcomm session, not this repository's NEON code. What is
shared is the *specification and the acceptance test*: identical D, paths,
census window, penalties, and the same golden hash `46470bd7a464469d`, held at
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

⭐ **The per-core reading is controlled rather than indicative: A78C 395 vs
A720 423 MDE/s — a 7% gap.** (Both re-verified against the corrected golden;
the A78C figures moved +1.2% and +2.6% from their pre-fix values.) Two wide out-of-order Arm
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
copy-bandwidth probe on the same part returns **249 GB/s**. The aggregate phase
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
| 1920×1080 | 6,266 | 10,077 | **12,260** |
| 1280×720 | 5,639 | 9,467 | 12,025 |
| 960×540 | 6,213 | 9,267 | 11,621 |
| 640×480 | 6,222 | 9,187 | 11,299 |

🔴 **This grid replaces one published with 12 timed frames per cell, which was
not enough sampling and produced a wrong number that a headline rested on.** The
640×480 D=64 cell read 5,108 MDE/s; at 60 frames it is **9,187 — the published
value was 80% wrong**, and it made D=128 appear *faster* than D=64 at the same
resolution, which is physically impossible and was quoted unremarked. Found by
an adversarial review, not by us.

⭐ **A larger disparity range is cheaper per disparity** — 6,266 → 10,077 →
12,260 MDE/s as D goes 32 → 64 → 128 at 1080p, a **+96%** gain. That finding is
robust and matches what the qualcomm session measured on a Hexagon NSP.

⚠️ **Efficiency does fall as resolution falls, but far less than we published.**
At D=64 the corrected sweep gives 10,077 → 9,467 → 9,267 → 9,187: a **1.10×**
fall from 1080p to VGA, not the **1.97×** that the under-sampled grid showed. The
*direction* still agrees with the Hexagon result; the *magnitude* does not — their corrected
figure is a 25.5% fall against our 8.8%. Claiming the two
"reproduce" each other was too strong. Lower resolution still buys latency rather
than throughput per disparity, but on this kernel the effect is small.

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

### One law: raising D amortises any per-pixel cost that does not scale with D

Three architectures show the same observation for three different reasons. The
qualcomm session proposed a lane-filling model, **predicted flat for CUDA, and
the measurement said +22%** — so the model was refuted on our hardware and the
corrected version, theirs, is more general than either of our originals:

> **Raising D amortises any per-pixel cost that does not scale with D.**

| | lanes | D=64→128 | the fixed cost being amortised |
|---|---|---|---|
| **NEON** (A720) | 16 B | **−2.4%, flat** | *none* — a vector is already full at D=64 |
| **HVX** (Hexagon) | 128 B | **+13.6%** | **idle lanes** — 64 of 128 doing nothing |
| **CUDA** (Thor) | warp covers D | **+22%** | **an irreducible cross-lane reduction** |

Lane-filling was the special case where the fixed cost happens to be wasted
lanes. On CUDA the lanes are never idle, yet a cost remains: every recurrence
step does a 5-step `__shfl_xor` min-reduction plus 2 neighbour shuffles — **7
shuffles per pixel per path, independent of D** — divided over DPT = D/32
disparities per lane.

⭐ **And on CUDA the effect reverses.** Measured on the shared scene below:

| D | ms | MDE/s |
|---|---|---|
| 64 | 13.19 | 10,062 |
| 128 | 21.86 | **12,138 — peak** |
| 256 | 56.41 | 9,409 |

The Hexagon rises monotonically through D=256; Thor turns over. That is what the
law predicts: the amortisation benefit saturates as 1/DPT while **S traffic grows
linearly with D** (1.06 GB at D=256), so past some D the traffic term wins.
**Which wall you hit is a property of the machine, not the algorithm** — Thor is
memory-bound at 145 of 249 GB/s, the Hexagon is still latency-bound at 6.8 of 28.

### 🔴 A prediction that was reported as confirmed, and is not

**Retracted 2026-08-29.** When the corrected grid weakened our resolution
penalty to 8.8% against the qualcomm session's 34%, we predicted the difference
was mechanistic: **the size of the resolution penalty should track how
latency-bound the part is.** They tested it within one chip — D=64 leaves the
128-byte HVX vector half empty and so more latency-exposed; D=128 fills it — and
reported it **confirmed**, 31.3% against 24.1%, the direction predicted.

**Both figures were single-invocation minima**, on a board they had established
in the same message was bimodal. Re-run over six invocations per cell, on
discriminating scenes, using medians:

| | 1080p → VGA | penalty |
|---|---|---|
| Hexagon D=64 | 893 → 665 MDE/s | **25.5%** |
| Hexagon D=128 | 1094 → 787 MDE/s | **28.1%** |

**The direction reverses.** And the reversal is not a finding either: 2.6 points
against per-cell spreads of 1.5–10.3% means the two are **indistinguishable at
this sampling**. The honest statement is that **the test does not resolve the
prediction in either direction — it was never powered to.**

⚠️ One weaker observation does point that way and is labelled rather than
substituted: the D=128 cells are *tighter* (spreads 1.5/1.9/9.1/2.3% against
D=64's 10.3/9.0/2.0/4.8%). Less run-to-run variance in a less latency-exposed
configuration is consistent with the mechanism — but variance is not the
quantity predicted, so it is a different and weaker claim.

🚨 **The estimator was the defect, and it is worth naming.** On multi-modal
hardware **the minimum is not a conservative choice** — it is the statistic most
sensitive to how many invocations happened to catch the fast mode, so it
silently rewards whichever cell got lucky. Their 1080p minimum was one fast
sample against five clustered 10% higher, and 1080p is the *numerator* of the
penalty, so that single sample propagated straight into the headline. Their
resolution penalty is therefore **25.5%, not the 31.3% published this morning**,
and ours should be compared against that.

⭐ **What survives is the cross-platform claim, which never rested on this.**
Hexagon 25.5% against Thor 8.8% is a 2.9× gap explained by the platforms —
latency-bound at 6.8 of 28 GB/s versus memory-bound at 145 of 249 — not by the
within-Hexagon D comparison. That stands on its own evidence.

### A shared scene, because a D-sweep can fail to discriminate

🚨 **A scene whose true disparities all fall below D cannot distinguish a correct
implementation from one that silently searches a shorter range.** That applies to
any D-sweep, ours included — and it already bit this project once, when twelve
`GOLDEN OK` lines came from a D=64 scene.

`data/shared/` is therefore a published cross-platform artefact: 1920×1080, seed
3, generated at D=256 semantics so its true disparities reach **254** and
truncate distinctly at 64, 128 and 256.

| artefact | sha256 (first 16) | reference hash |
|---|---|---|
| `left.pgm` | `99b2a43de9d6dc82` | — |
| `right.pgm` | `f8cc283345e4a596` | — |
| `golden_d64.pgm` | `9c4ff74807b3b537` | `004d85c8288ef5b6` |
| `golden_d128.pgm` | `fb4b612656137745` | `760646cc7238afea` |
| `golden_d256.pgm` | `ae8df5c081dde351` | `f4d47de16696e1b7` |

Goldens are from the x86-64 scalar reference; the CUDA implementation reproduces
all three on Thor. Three distinct hashes at three D values is the property that
makes the sweep a test rather than a formality.

### Why D gets cheaper per disparity — the CUDA mechanism in detail

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

### The discrete 5090's stereo engine: already removed, and the fallback costs

Both Jetsons expose a fixed-function stereo engine. The discrete RTX 5090 has
the same accelerator family (NVOFA) — its driver library exports
`NvOFAPICreateInstanceCuda` and references `NV_OF_INIT_PARAMS::disparityRange` —
so it was measured directly through the public Optical Flow SDK headers.

⭐ **`NV_OF_MODE_STEREODISPARITY` fails on the 5090 with
`NV_OF_ERR_UNSUPPORTED_FEATURE`** at every output grid size (1, 2 and 4).
General optical-flow mode works on the same device, so this is mode removal
rather than a broken engine. **NVIDIA documents the deprecation in future tense;
on Blackwell consumer silicon it has already happened.**

The documented fallback is to read the X component of general optical flow —
disparity *is* the X component, since a rectified pair's matches are horizontal
by construction. Measured, 1920×1080, 30 timed iterations, same scene and mask:

| | time | bad > 1px | MAE |
|---|---|---|---|
| NVOFA general optical flow (2D search) | 8.50 ms | 5.3% | 5.59 |
| **our CUDA SGM, D=128 (1D search)** | **4.13 ms** | **2.8%** | **0.48** |

On this part **the software is 2.06× faster and substantially more accurate than
the hardware engine's fallback path** — while doing *less* work, a 1D epipolar
search against a 2D one. That asymmetry is the result, not a caveat.

🚨 **And the fallback's cost is measurable rather than theoretical.** Stereo mode
constrains the search to the epipolar line and therefore *cannot* return a
vertical match. General optical flow can, and on a rectified pair any non-zero
Δy is by definition an error: **13.4% of pixels have non-zero Δy, the largest
75 pixels.** Those are errors the removed mode was structurally incapable of
producing — the deprecation trades a guarantee for generality.

⚠️ **Not bit-exact and not gated.** NVOFA is NVIDIA's implementation with its own
cost function; this is a throughput and accuracy comparison, not the same test.
⚠️ An earlier draft said these headers were "behind an NVIDIA developer login".
**That was wrong** — they are public at `github.com/NVIDIA/NVIDIAOpticalFlowSDK`,
and the claim came from guessing two repo names and reading the 404s as a
finding.

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

### Is throughput just bandwidth? Measured on all three GPUs

The kernel is memory-bound, so the honest cross-GPU question is whether MDE/s is
merely a proxy for DRAM bandwidth. Ceilings measured with a streaming-copy probe
on each part, achieved figures from the traffic model against the aggregate
phase:

| GPU | aggregate | achieved | ceiling | utilisation | MDE/s | MDE/s per GB/s |
|---|---|---|---|---|---|---|
| RTX 5090 | 2.49 ms | 587 GB/s | 1,385 GB/s | 42% | 37,383 | 27.0 |
| Thor | 10.04 ms | 145 GB/s | 249 GB/s | 58% | 10,062 | 40.5 |
| Orin AGX | 16.83 ms | 87 GB/s | 175 GB/s | 49% | 5,701 | 32.5 |

**Largely yes.** Across a 7.9× spread in bandwidth, utilisation stays in a narrow
42–58% band and throughput-per-unit-bandwidth varies by only 1.5×. Bandwidth
predicts most of the ranking.

⭐ **The residual is the interesting part: the 5090 is the *least* efficient of
the three despite being the fastest.** At 1,385 GB/s the kernel can no longer
saturate the memory system, so on that part it has stopped being purely
bandwidth-bound — the same transition qualcomm observed in reverse on the
Hexagon, which is latency-bound at 6.8 of 28 GB/s. Which wall you hit is a
property of the machine, and these three GPUs sit at different points on it.

⚠️ **This was a gap until asked about.** Bandwidth had been measured on exactly
one board (Thor) and used to justify a claim about all of them. The ceilings for
Orin and the 5090 did not exist until someone asked whether they had been
recorded. A memory-bound conclusion resting on one machine's memory measurement
is thinner than it looked.

## 🔴 Real imagery does not by itself retire the discrimination check

The first Middlebury scene here was **not a valid correctness gate**, and
`check_golden_discriminates.py` passed it. Found by the qualcomm session, who
turned the check into a script and swept every (scene, D) pair either corpus
cites.

At 1/3 scale the Motorcycle pair's true disparities stop at **79.9** while the
test ran at **D=128**. The golden reached 127 — 126 distinct values, apparently
healthy — but only **58 pixels (0.0088%)** sat in the top decile, and all 906
pixels above the true maximum were the implementation's own **errors**.

⚠️ **And the failure mode gets worse as the implementation gets better.** If a
golden only reaches the top of its range through high-disparity mistakes, a
*more accurate* implementation makes fewer of them and becomes **harder** to
distinguish from a truncated one. A test whose power decays as the thing it
tests improves is worse than no test.

**Two fixes, because there were two faults:**

- **The checker** now returns BLIND / THIN / OK rather than pass/fail. Where
  `gt_float.npy` sits beside a golden it separates *true* high-disparity pixels
  from the implementation's errors, and fails a golden whose top of range is
  reached only by the latter. Top-decile coverage threshold 0.5%.
- **The scene was mismatched to D**, which was the root cause. Regenerated at
  1/2 scale — 1482×1000, true disparity max 119.8 against D=128 — giving
  **1.93% top-decile coverage of which 22,530 pixels are genuinely high**,
  not 58 errors.

| target | ms | spread | golden `e8a95242882013f0` |
|---|---|---|---|
| RTX 5090 | 3.32 | 1.16 | ✓ |
| Thor | 15.94 | 1.02 | ✓ |
| Orin AGX | 23.91 | 1.02 | ✓ |
| 8× A720 | 112.44 | 1.07 | ✓ |
| 6× A55 | 590.50 | 1.00 | ✓ |

Accuracy against dense ground truth: **bad>1px 16.2%, bad>2px 11.2%, MAE 3.35**
— higher than the mismatched scene reported, because D=128 is now actually being
exercised.

### The accuracy figures were reproduced independently

Before the scene was corrected, the qualcomm session ran their HVX kernel on it
and **scored it themselves against `gt_float.npy` rather than citing ours**:

| | ours | theirs, independently derived |
|---|---|---|
| bad > 1px | 13.9% | **13.95%** |
| bad > 2px | 10.2% | **10.24%** |
| MAE | 2.49 | **2.49** |

557,252 pixels scored, leftmost 128 columns excluded. Their kernel also
reproduced the golden byte-for-byte — 0 differing pixels of 658,008 — making the
Hexagon a **sixth target** on that scene at 86.13 ms.

⭐ **This is the first accuracy number in the project to come from real
calibrated capture, and it was confirmed by a different implementation on
silicon we have never touched.** They declined to send their disparity map on
the grounds that a matching hash means it *is* our map — the pixels only carry
information the hash cannot when a hash disagrees.

⚠️ Those figures stand as accuracy; the run predates the scene correction above,
so it is not comparable to the 1482×1000 timings and is not in the main table.

⭐ **The general form, and it is the qualcomm session's: discrimination is a
property of the (SCENE, D) PAIR** — not of the scene, the generator, or the
algorithm. Neither "we use real imagery now" nor "we fixed the generator"
retires the check. It has to run per pair, which is why it has an exit code.

## Real imagery, and the last unverified rows

⭐ **Every published row is now verified against a golden that discriminates.**
The two Mali rows were the last resting on the old blind golden; both re-run:

| | published | re-verified | spread |
|---|---|---|---|
| Mali-G720 (10 CU) | 256 ms | **257.13 ms** ✓ | 1.01 |
| Mali-G310 (1 CU) | 1846 ms | **1848.96 ms** ✓ | 1.00 |

Unchanged within 0.4%, both `GOLDEN OK` on `46470bd7a464469d`.

⚠️ The G310 build needed `/usr/lib/libOpenCL.so.1` linked directly: the `.so`
symlink a linker looks for does not exist on that image. That is the same shape
as the "no compute driver" myth this report already documents — the runtime is
present and working, and only a missing development symlink makes it look absent.

### The cost gate is now armed on seven configurations

It was armed on three, and not on the tuned kernel at all. Now:

    a55 imx95 988x666 D128        mali_cl imx95 1920x1080 D64
    a55 o6    988x666 D128        mali_cl o6    1920x1080 D64
    cuda_opt thor / orin / rtx5090  988x666 D128

⭐ **And the calibration is diagnostic even where it passes.** The G720's argmin
runs at 0.654 ns/op against census at 0.064 — **ten times less efficient than the
cheapest phase in the same kernel**, which is the serial 64-iteration argmin
scan, one work-item per pixel. The same bottleneck the CUDA port had before it
was rewritten as a warp reduction. The gate does not fire, because it is
calibrated against itself, but the numbers name the phase to fix.

## Measurement quality: dispersion, transfers, and observed thread counts

Three instrument defects were fixed after review. All targets were then
re-measured against the corrected goldens.

### Every target, re-verified with its spread

| target | ms | p95/median | min/median | golden |
|---|---|---|---|---|
| 6× Cortex-A55 | 345.55 | 1.01 | 0.99 | `46470bd7…` ✓ |
| 1× Cortex-A55 | 1595.95 | 1.00 | 1.00 | ✓ |
| 8× Cortex-A720 | 50.80 | 1.01 | 1.00 | ✓ |
| 1× Cortex-A720 | 316.34 | 1.00 | 1.00 | ✓ |
| Thor D=64 | 13.16 | 1.04 | 0.99 | `4518557a…` ✓ |
| Thor D=128 | 21.56 | 1.04 | 0.99 | `3d99f1c7…` ✓ |
| Orin D=64 | 23.90 | 1.01 | 0.97 | ✓ |
| Orin D=128 | 33.41 | 1.01 | 0.98 | ✓ |
| **RTX 5090 D=64** | 3.43 | **1.12–1.29** | 0.92 | ✓ |
| RTX 5090 D=128 | 4.23 | 1.19 | 0.92 | ✓ |
| RTX 5090 D=256 | 6.91 | 1.11 | 0.92 | ✓ |

🔴 **Locking the clock was tried and made it worse — the drift hypothesis is
falsified.** The obvious explanation for the 5090's spread was boost-clock drift,
so the SM clock was locked at 2595 MHz and held perfectly: 24 of 24 samples taken
*during* a run read exactly 2595, with no power-cap or slowdown events. The
spread did not improve:

| D=256, same protocol | median | within-run spread | between-run |
|---|---|---|---|
| unlocked | **6.88–6.91 ms** | 1.11 | ±0.4% |
| locked at 2595 MHz | 7.14–7.22 ms | 1.16 | ±0.6% |

Locking cost **4.3% performance** — 2595 caps a boost the card genuinely reaches
under load — and left the spread slightly *worse*, reproducibly over five runs.
**So the variance is not clock drift**: it survives a rock-steady clock. What
remains is kernel-launch jitter and the co-resident process. The lock was
reverted; the unlocked figures are the ones published.

🚨 **The 5090 is the only unstable platform, and it is the only shared one.**
Its clocks are unlocked (2610 of 3105 MHz, no permission to lock without root)
and a `rustdesk` process co-resides on the card. A longer warm-up does not help:
spread holds at ~1.12 whether warming 10, 100 or 300 frames. Between-run medians
*are* reproducible — D=64 spans 3.26–3.48 over five runs, D=256 only 6.88–6.91 —
so the figures are sound but carry more uncertainty than any dedicated board.
Every dedicated board sits at 1.00–1.04.

**The harness now computes p95/median and min/median on every run and prints
`UNSTABLE` past a threshold.** No dispersion figure appeared anywhere in this
project until 2026-08-29, which is exactly how a cell that was 80% wrong reached
publication carrying p95/median 1.55.

### Host↔device transfer was folded into the phase timings

`census_ms` included both H2D copies and `argmin_ms` included the D2H copy, so
every per-phase claim derived from them was phase-plus-copy. Now separated:

| | transfer | share of total |
|---|---|---|
| Thor D=64 | 0.126 ms | 1.0% |
| Orin D=64 | 1.564 ms | **6.5%** |

⚠️ **This corrects a published figure.** "Orin's argmin is 3.9× less efficient
than census" was computed with transfers inside both. Excluding them the ratio is
**4.59×** — census shed two H2D copies while argmin shed only one D2H, so the
published number *understated* the imbalance.

### The `-t` fix reported a request, not an observation

The earlier fix reported `omp_get_max_threads()`, which echoes the requested
count back through the runtime rather than observing the team that ran — the same
failure shape as the inert flag it replaced. Implementations now report
`omp_get_num_threads()` from **inside** the parallel region, and that observed
count is what reaches the table and the JSON.

## 🔴 A second independent review, and the off-by-one that survived the first fix

A final adversarial review of the finished repo found the acceptance-model bug
had been fixed **one step short of where the mechanism goes** — a pattern it
identified across several fixes.

**The generator pinned its top slab to `SGM_D−2`, and the checker was set to
`D−2` to match.** So `d = D−1` — the lane where *every* implementation
special-cases "d+1 out of range, treat as 255", i.e. the single most bug-prone
disparity in the range — was unreachable: the D=256 golden had **zero** pixels
there, the D=64 golden had **two**. An implementation that never searched the
top disparity reproduced them byte-for-byte, and the checker printed PASS.
The checker had been weakened to make the generator's output pass, instead of
the generator being fixed to satisfy the checker.

Fixed at the root this time: the generator pins to `D−1`, the checker requires
`D−1`, and the regenerated goldens have the top disparity winning at scale —
218,075 pixels at d=63, 84,720 at d=255. **All six local targets re-verified**
on the new primary golden `0bc0102058d1505f`.

The same review also found: the real-imagery golden's top decile "of which"
count was set arithmetic done wrong (gt≥115 counted over the whole image, not
intersected with the golden — 20,024, not 22,530); the roofline calibrations
referenced a scene that no longer existed, so the gate could never arm on any
live configuration (recalibrated at 1920×1080 D=64 on all five targets, and
`sweep.sh` no longer passes `--no-roofline`); the sweep's documented build-cmd
argument was dead code and a failed generator build silently reused a stale
binary (both abort now); a latent out-of-bounds read in `store_plane` at
SGM_D=16; the p95 index returning the maximum for n≤20 while labelled "p95";
and roughly a dozen places where a corrected number's stale copy survived in
another section. All fixed; the stale-copy sweep is the repo's own value-diff
rule, which had not been applied to itself.

## The original acceptance-model finding

### The first finding: a golden blind to the top third of the range

**Until 2026-08-29 the primary golden could not discriminate the top third of the
disparity range.** `data/golden/synthetic.pgm` had a maximum disparity of 44 with
`SGM_D=64`, so disparities 45–63 never won anywhere in it. The consequence,
proven by an independent numpy re-implementation and confirmed here:

> **An implementation that only searched `d < 45` reproduced the golden
> byte-for-byte.**

Every target accepted against that golden — A55, A720, A78C, both Malis, the
Hexagon — had the top 30% of its disparity range verified by nothing. Only the
CUDA implementation had also been run against the `data/shared` goldens.

**The root cause was structural, not a bad seed.** `gen_synthetic.c` assigned
slab disparity as `SGM_D/4 + rand % (SGM_D/2)`, whose maximum is `3D/4 − 1` — 47
at D=64. The generator could never populate the top quarter of *any* range, and
the `if (d > SGM_D-2)` clamp that followed was dead code that made the bound look
deliberate. The shared D=256 scene previously topped out at 187 for the same reason.

⚠️ **This project named this exact failure mode, then shipped it.** "A check that
cannot discriminate" is documented below as one of four ways a gate reads green
while being worthless. It was fixed for the D-sweep scene and never checked for
in the primary golden — and `max disparity present = 44` was actually *measured*
during the sweep work and read as a build-flag problem rather than an
acceptance-model one. **Having a name for a failure is not the same as having
checked for it.**

### What was done about it

- **The generator now spans the full range.** Slab disparities are stratified
  across `bgd+1 .. SGM_D-2` in bands, with the last slab pinned to the maximum,
  so the top is populated by construction rather than by luck of the seed. Slab
  count raised 6 → 12.
- **`scripts/check_golden_discriminates.py` asserts the property**, so this
  cannot silently recur. It fails the old golden and passes the new ones.
- **Every golden was regenerated and every reachable target re-verified.**

| golden | old hash | new hash | top disparity |
|---|---|---|---|
| primary D=64 | `b1b407b5949f0cc1` | **`46470bd7a464469d`** | 44 → **63** |
| shared D=64 | `d98d7b0718e2f8b9` | **`4518557a40d1b500`** | → 63 |
| shared D=128 | `97da516d22851e0c` | **`3d99f1c7e392c712`** | → 127 |
| shared D=256 | `cb492999e5700840` | **`6c38c42dc67b4d33`** | 187 → **254** |

⭐ **Every implementation passed the corrected goldens, with timings unchanged
within noise** — A55 342.76 ms (was 341.3), Thor 13.27 (13.19), Orin 23.51
(23.28), 5090 3.32 (3.55). So the implementations were right all along and no
published timing needed revision. **Only the evidence was weak** — which is
precisely the situation a gate exists to prevent you from being in unknowingly.

### The other things the review found

- **`sweep.sh` swallowed `GOLDEN MISMATCH`** with `|| true`, so a wrong-answer
  cell landed in the results JSON. It now aborts.
- **`geometry.py` ingested every row with no `golden_match` filter**, so a failed
  cell's timing flowed into the published grid and every DERIVED range/FOV figure
  computed from it. It now refuses unverified rows.
- **The cost gate was never armed on the tuned kernel.** It registers as
  `cuda_opt`; the calibration file held `cuda` entries from the *naive* kernel,
  so no key matched. `sweep.sh` passed `--no-roofline` outright, and `make check`
  defaults `BOARD=unknown`. "On by default" described a code path, not the
  published measurements.
- **The calibration key ignored resolution and D**, while the repo's own data
  shows ns/op moving 22% with D — so any legitimate configuration change either
  false-failed or forced `--no-roofline`. The key now includes both.
- **The roofline failed *open* on a missing aggregate phase**, leaving ~94% of the
  work ungated while printing a green line. It now fails closed.
- **`pin.sh` could not pin the A78C at all** — it maps `a78 → 0xd41`, but the
  A78C's MIDR part is `0xd4b`. The "always pin" methodology cannot have produced
  the A78C rows with this tool. Added.

## 🔍 Adversarial review of these results

Written by attacking the results rather than restating them. Five findings, in
descending order of how much they should change your reading.

### 1. 🔴 The OFA accuracy comparison does not work, so no quality claim is made

The throughput comparison against NVIDIA's hardware engine says **nothing about
output quality**, and an attempt to measure that failed:

| | bad > 1px | MAE |
|---|---|---|
| our SGM, restricted to the searchable range | **2.6%** | **0.45** |
| OFA, same restriction, best of three descalings | 100% | 21.2 |

Our own implementation validates beautifully against the synthetic ground truth,
which proves the method and the GT are sound. OFA's output does not align under
any simple scaling — descaled mean 52 against a GT mean of 34 — so either its
disparity convention differs from ours or my extraction from the sample's
display-scaled PNG is wrong. **Until that is resolved, OFA's accuracy is
unmeasured and the 2.34× is a throughput number only.**

⚠️ **My first attempt at this measurement was invalid and would have been
embarrassing.** Comparing over the whole frame gave 40.4% bad pixels for our own
bit-exact implementation. The cause: 38.8% of the scene has true disparity ≥ 120
and cannot be found at D=128 at all. Scoring an algorithm on pixels it is
structurally incapable of reaching is not a measurement of the algorithm.

### 2. ✅ The path-count mismatch was real and turned out not to matter

VPI defaults to `includeDiagonals = 1` — **8 paths against our 4** — so the
comparison was between unequal amounts of work. Rather than apply the
literature's 1.45× scaling, it was measured:

| | 8 paths | 4 paths |
|---|---|---|
| Thor OFA | 9.42 ms | 9.85 ms |
| Orin OFA | 72.66 ms | 72.45 ms |

**OFA's cost is independent of path count on both parts.** The timing comparison
therefore stands unchanged — and the correct reading is stronger than before:
OFA delivers an *8-path* result in the time our kernel delivers a *4-path* one.

### 3. ⚠️ The timed regions are not identical

Our figure is end-to-end: host→device copy, census, cost, aggregate, argmin,
device→host copy. The OFA figure wraps only `vpiSubmitStereoDisparityEstimator`
plus the stream sync; the sample's colour conversion and any upload sit outside
it. The excluded work is sub-millisecond against 9–22 ms, so this biases in
OFA's favour by under ~5% and does not change the conclusion — but the two
numbers are not the same measurement and should not be quoted as if they were.

### 4. ⚠️ The 249 GB/s ceiling is a best case, so "59% of it" understates us

That figure comes from a pure `float4` streaming copy. Our aggregate phase is a
read-modify-write over a 265 MB working set with a half-width scratch — a
pattern that cannot reach copy bandwidth on any GPU. The true achievable ceiling
for this access pattern is lower and unmeasured, so 145 GB/s is closer to the
real limit than "59%" suggests.

### 5. ⚠️ Two comparisons rest on things that are not known

- **The A78C per-core result assumes comparable clocks.** "A78C 402 vs A720 423,
  a 5% gap" is only a statement about cores if they run at similar frequencies.
  The A720 figure is at a measured 2.5 GHz; the A78C clock was never read,
  because that board is under another session's lease. If the A78C is slower per
  clock the conclusion inverts.
- **"699× the scalar reference" is not a speedup claim.** That reference is
  deliberately unoptimised — `-O2`, no SIMD, single thread — and exists to define
  the golden, not to be competitive. It is the floor, not a baseline.

Also: **at the time of that review, every timing came from one synthetic scene; a real capture has since been added.** SGM's
control flow is data-independent so the numbers should hold, but that is an
argument, not a measurement.

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
i.MX 95 FRDM, `OMP_NUM_THREADS` unset, golden `46470bd7a464469d` on all four):

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

- **libSGM comparison on the RTX 5090** not run (our own kernel is measured
  there; the third-party CUDA implementation is not).
- **Power** unmeasured on every platform; the perf/W column cannot be filled
  from these boards.
- **8-path** numbers not taken; all results are 4-path. The literature's 8→4
  reduction is ×1.45 for +0.2% error (SOURCED), so an 8-path figure would be
  roughly 0.69× these.
- **A gcc-13+ A720 rebuild** could move the headline number in either direction.
- **Real-imagery timings** exist for one scene (Middlebury Motorcycle); the
  headline table remains the synthetic configuration.
