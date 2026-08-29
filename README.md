# sgm-bench

**How fast can a CPU do Semi-Global Matching, if you refuse to let it be wrong?**

A from-scratch SGM stereo implementation benchmarked across nine execution
targets — x86-64, Cortex-A55, Cortex-A720, Cortex-A78C, two Mali GPUs via
OpenCL, a Hexagon NSP via FastRPC, and Jetson Orin and Thor via CUDA — where
**every target must produce the same output, byte for byte**, or its timing is
thrown away.

Current best: **2,569 MDE/s at 1920×1080 on eight Cortex-A720 cores**, which is
10.4× the fastest published Arm-CPU SGM at matched disparity range and matched
path count. See [`REPORT.md`](REPORT.md) for the full measurement record,
including the five optimisations that were tried and rejected.

---

## The one design decision that matters

**Timing and correctness are checked in the same run, and correctness gates.**

    $ ./bin/sgm_a55 left.pgm right.pgm -g golden.pgm
    a55  1920x1080 D=64 paths=4 th=6  median 342.75 ms  ...  hash b1b407b5949f0cc1  GOLDEN OK

The harness computes an FNV-1a hash of the disparity map, compares against a
golden map produced by a deliberately boring scalar reference, and **exits 2 on
mismatch**. There is no mode in which you get a number without the check.

This is not ceremony. SGM optimisation fails in a specific, nasty way: the fast
broken version looks fine. A wrong disparity map is still a plausible-looking
grey gradient — no crash, no NaNs, no zeroes, nothing a smoke test catches. Two
separate incidents in this project were caught *only* by the hash:

- a Hexagon kernel that was **3× faster with the wrong answer**, because HVX
  widening is deinterleaved (the low vector holds even-indexed bytes, not bytes
  0..63), so costs were accumulated onto the wrong disparities;
- a stale golden file on one machine that would have silently failed every
  correct implementation pushed to it.

If you take one thing from this repo, take this: **an optimisation harness that
reports time without simultaneously proving output is a harness that will
eventually publish a lie.**

---

## Results

MEASURED, 1920×1080, D=64, 4 paths, 9×7 census. Full provenance in `REPORT.md`.

| target | class | threads | ms | **MDE/s** | per-core |
|---|---|---|---|---|---|
| **NVIDIA Thor GPU** (Blackwell, 20 SM, CUDA) | GPU | — | 13.2 | **10,062** | — |
| **NVIDIA Orin AGX GPU** (Ampere, 16 SM, CUDA) | GPU | — | 23.3 | **5,700** | — |
| 8× Cortex-A720 @2.2–2.5 GHz (Radxa O6) | CPU | 8 | 51.7 | **2,569** | 423 |
| 8× Cortex-A78C (IQ-9075) | CPU | 6 | 94.7 | 1,402 | 402 |
| **Hexagon v73 NSP** @1.46 GHz | DSP | 4 | 138.6 | **957** | — |
| Mali-G720, 10 CU | GPU | — | 256.0 | 518 | — |
| 1× Cortex-A720 @2.5 GHz | CPU | 1 | 313.6 | 423 | 423 |
| 1× Cortex-A78C | CPU | 1 | 332.2 | 400 | 400 |
| 6× Cortex-A55 @1.8 GHz (i.MX 95) | CPU | 6 | 341.3 | 389 | 84 |
| 1× Cortex-A55 @1.8 GHz | CPU | 1 | 1588.7 | 84 | 84 |
| Mali-G310, 1 CU | GPU | — | 1846.0 | 72 | — |
| scalar reference, 1× A55 | floor | 1 | 9188.0 | 14.4 | — |
| *NVIDIA RTX 5090 GPU* | GPU | — | *not measured* | *—* | — |

Thor's peak across the D sweep is **12,138 MDE/s at D=128**; it falls back to
9,409 at D=256, where S traffic (1.06 GB) overtakes the amortisation gain.

**MDE/s** = million disparity estimations/sec = `W × H × D / runtime`. It is the
unit the literature uses because fps hides D — two implementations can both
report "30 fps" while doing 100× different work. It normalises D but **not**
census window size or path count, so compare those explicitly.

The most interesting row is the DSP, and not for the reason first published
here. SGM's aggregation is a serial recurrence in x or y, so the obvious reading
is that a wide unit can fill lanes but cannot hide the dependency. On that
reading the Hexagon NSP first measured **18.7× slower** than the CPU cluster.

It is now **1.45×**, and it beats a *single* A78C core by **2.40×** — a 13× move
in one day, bit-exact throughout. **The latency is hideable**, by interleaving
independent dependency chains at a cost in registers; the part that was
latency-bound was never bandwidth-bound (28.17 GB/s available, 6.8 consumed).

⚠️ Which means the Mali figure here is probably a floor too, by the same
mechanism: `mali_cl/sgm.cl` carries **one** dependency chain per work-item with
three barrier sites per step and never interleaves. 518 MDE/s is what this
kernel does, not what the G720 can do.

Read that as *weaker* than it sounds, not stronger. The mechanism predicts the
number is low; it does not predict **by how much**. 518 is now an **unbounded
floor** rather than a measurement of the hardware, and "the GPU could be much
faster" is not the same claim as "the GPU is competitive". Nobody has done for
the Mali what was done for the DSP, so nobody knows. `REPORT.md` has the full
trajectory, including a DERIVED ceiling that was wrong by 4×.

---

## Quick start

Needs gcc with OpenMP and NEON. On the target board:

    make MCPU=cortex-a55 golden   # build reference, synthesise a stereo pair, make the golden map
    make MCPU=cortex-a55          # build the implementations
    make MCPU=cortex-a55 check    # every impl vs golden; exits 2 on any mismatch

Then benchmark, pinned:

    ./scripts/pin.sh              # run directly: prints CPU topology by MIDR part

    source scripts/pin.sh        # then SOURCE it — pin_run is a shell function
    pin_run a55 6 ./bin/sgm_a55 \
        data/synthetic/left.pgm data/synthetic/right.pgm \
        -g data/golden/synthetic.pgm -t 6 -j results/run.json --board imx95

`pin_run <uarch> <nthreads> <cmd...>` sets `OMP_NUM_THREADS`, `OMP_PLACES=cores`,
`OMP_PROC_BIND=close` and a `taskset` mask, forces the performance governor, and
logs per-core frequency before and after the run so you can see throttling.

It selects cores by **MIDR part number, not CPU index** (`pin_list a720` →
`4,5,6,7`), because CPU numbering is not stable across boards and "the big
cluster" is not a portable concept. Use it rather than invoking the binary bare:
on a heterogeneous SoC the scheduler will otherwise migrate threads onto the
wrong cluster mid-measurement.

Harness flags: `-o out.pgm`, `-g golden.pgm`, `-t threads`, `-w warmup`,
`-n timed`, `-j results.json`, `--board name`. Timing is the median of `-n`
runs; p95 and min are also recorded.

---

## Why SGM is hard

Stereo matching is easy to state and awkward to make fast. For every pixel you
score **D** candidate disparities, then smooth those scores so the depth map is
coherent rather than speckled. The smoothing is what makes it good, and it is
also what makes it hard:

    L(p, d) = C(p, d) + min( L(p−1, d),  L(p−1, d±1) + P1,  min_k L(p−1, k) + P2 ) − min_k L(p−1, k)

**Every pixel depends on the pixel before it.** That is a serial recurrence along
each of four sweep directions, so the obvious parallelism — one thread per pixel —
is unavailable. What is left is parallelism across disparities and across
independent scanlines, and a cross-lane `min` over all D at *every step*.

Three properties fall out, and they shape every implementation here:

- **The work is large and fixed.** 1920×1080 × 64 disparities × 4 paths ≈ 530M
  updates per frame. There is no early exit and no data-dependent shortcut.
- **The intermediate is bigger than the image.** The aggregated cost volume is
  265 MB per frame at 1080p — 128× the input — so a tuned implementation ends up
  **memory-bound**, not compute-bound. Ours reaches 145 GB/s of a measured
  247 GB/s ceiling on Thor.
- **Wide hardware does not automatically win.** A vector unit can fill lanes but
  cannot remove the dependency; hiding it takes independent chains and costs
  registers. That is why the ordering below is not the ordering of raw FLOPS.

![SGM throughput by processing unit](docs/results.png)

Every bar is bit-exact to the same golden hash. The scalar reference at the
bottom is the floor the whole exercise is measured against — the fastest bar is
**700× faster than it, computing byte-identical output**.

## There is dedicated stereo hardware on these parts, and we measured it

Both Jetsons carry an **OFA — Optical Flow Accelerator** — whose VPI header
documents a *"semi-global matching (SGM) computation"* with optional diagonal
paths. It is a fixed-function stereo engine, exposed as
`VPI_BACKEND_OFA`, and it only accepts `maxDisparity` of **128 or 256**.

![hardware SGM vs our software](docs/hardware-vs-software.png)

| board | our CUDA (bit-exact) | OFA hardware engine | |
|---|---|---|---|
| **Thor** | 21.86 ms · 12,142 MDE/s | **9.35 ms · 28,402 MDE/s** | hardware wins **2.34×** |
| **Orin AGX** | **33.16 ms · 8,005 MDE/s** | 72.64 ms · 3,654 MDE/s | software wins **2.19×** |

⭐ **The dedicated engine wins on one part and loses on the other.** Thor's OFA
is 7.8× faster than Orin's at the same settings — a generational jump in the
fixed-function block that is far larger than the 1.5× between the two GPUs. On
Orin, a hand-written CUDA kernel beats the silicon built for this exact job.

⚠️ **These rows are NOT bit-exact to our golden and are not gated by it.** VPI's
SGM is a different implementation — its own census, penalties, and confidence
output — so this is a throughput comparison at matched resolution and disparity
range, not the same test the rest of this repo runs. It is kept in a separate
figure for that reason.

⚠️ **Watch the downscale factor.** OFA defaults to `downscaleFactor=2`, which
emits a **960×540** disparity map from a 1920×1080 input and runs in 4.14 ms on
Thor. Quoting that as a 1080p result would overstate it by 2.3×. Every number
above is `downscaleFactor=1`, full-resolution output. This is the same
convention trap that made a vendor's quoted figure unusable earlier in this
project, met again in different silicon.

## The second gate: a predicted cost

The golden hash catches wrong answers. It is structurally blind to a kernel that
is **correct and slow** — and this project shipped only that gate for a full day,
during which two separate half-scalar kernels passed it with perfect output.

So there is a second gate, on by default, running in the same invocation as the
timing for the same reason the hash check does:

    make MCPU=cortex-a55 BOARD=imx95 roofline-cal   # record what "good" costs
    make MCPU=cortex-a55 BOARD=imx95 check          # gate every run against it

Each phase has a scalar-equivalent op count fixed by the workload (census is
`2·W·H·62` compares, aggregate is `PATHS·W·H·D·4`, and so on). Measured time
divided by op count gives **ns per op — an efficiency with a fixed denominator**,
which a share can never be. Each phase is then checked against its own calibrated
budget.

    roofline:  census 0.0838/0.0825 1.02x  aggregate 0.1322/0.1325 1.00x

Exit codes are distinct because the failures are: **2 = wrong answer**,
**3 = right answer produced too slowly**.

⚠️ **It was first built as a spread check between phases, and that does not
work.** Comparing phases to each other needs no calibration, which is appealing,
but it cannot catch a regression in the *fastest* phase: slowing the best phase
moves it toward the worst and the spread **shrinks**. Measured, by planting the
real bug — healthy spread 1.62×, planted spread 1.26×. The defect made the gate
look healthier.

**The threshold was picked by planting a defect, not by taste.** Five repeats of
a healthy A55 build hold census to 0.0819–0.0832 ns/op (1.6% spread). Planting a
half-scalar census — vector loop stopped 124 columns short, handing them to the
scalar tail — measures 1.25× its calibration. The limit is 1.15: ~15× above the
noise, below the defect, and **verified to fire**, exiting 3 with `GOLDEN OK`.

A gate you have never seen go red is not evidence. If you change the model,
plant something and watch it fail before you trust it again.

Two honest limits: the calibration records what a build *someone believed was
good* cost, so it catches regression from that point rather than a bad
implementation overall; and it is per implementation and per board, so on an
uncalibrated board it prints `ROOFLINE NOT ARMED` rather than passing silently.

## The workload

All parameters live in exactly one place, [`common/sgm_params.h`](common/sgm_params.h),
which no implementation may bypass. Defaults:

| | |
|---|---|
| disparity range `SGM_D` | 64 (multiple of 16; 128 also tested) |
| paths `SGM_PATHS` | 4 — L, R, U, D |
| census window | 9×7 → **62-bit** descriptor (centre excluded) |
| penalties | `P1=20`, `P2=192` |
| output | uint8 integer disparity; no LR check, no subpixel, no median |

The recurrence, per path:

    L(p,d) = C(p,d) + min( Lp(d), Lp(d±1)+P1, min_k Lp(k)+P2 ) − min_k Lp(k)

with `Lp` the previous pixel along the path, additions **saturating** at 255 and
out-of-range `d±1` treated as 255. The renormalisation subtraction is what keeps
`L` in 8 bits; `sgm_params.h` static-asserts `SGM_COST_MAX + SGM_P2 <= 255`,
which is the constraint that makes a uint8 path cost legal at all.

Ties in the final `argmin` go to the **lowest** d. That is an arbitrary choice,
but it must be the same arbitrary choice everywhere or the hash diverges on
perfectly valid implementations.

⚠️ **Census bit order is a free parameter and the spec pins it anyway.** Hamming
distance is `popcount(a ^ b)`, so any permutation applied consistently to both
descriptors leaves every cost unchanged. The order is fixed (dy outer, dx inner,
centre skipped, edges replicated) purely so independent ports land on the same
hash. If a permutation is cheaper on your ISA, take it — you are free here, and
the freedom is worth more as permission not to worry than as an optimisation.

---

## Layout

    common/          reference impl, harness, params, shared NEON census/hamming
    a55/             the main NEON implementation (also used on A720/A78C)
    a55/baseline/    phase-0 standalone benchmarks, kept unchanged for reference
    a55/sgm_fused.c  REJECTED wavefront experiment, kept for its header
    a720/            A720-specific variant (K-row interleaving; measured no gain)
    colmajor/        column-major layout experiment — contributed, not my work
    hvx/             Hexagon HVX kernels — contributed, not my work
    mali_cl/         OpenCL kernel + host
    cuda/            CUDA port of the OpenCL kernel (Jetson Orin / Thor)
    scripts/pin.sh   topology-aware pinning wrapper
    data/golden/     golden disparity map + sha256 + the params that produced it
    REPORT.md        the measurement record, with provenance on every number

`colmajor/` and `hvx/` were contributed by a collaborating session working on
Qualcomm silicon and are credited as theirs.

### Adding an implementation

Provide a translation unit exporting one symbol:

    const sgm_impl SGM_IMPL = { "myimpl", my_run };

where `my_run(left, right, W, H, disp, threads, stage_times)` fills `disp`. Then
in the Makefile:

    IMPLS += myimpl
    IMPL_myimpl_SRCS := myimpl/sgm_myimpl.c common/census_neon.c

`make check` picks it up automatically. The Makefile ships with only `a55`
enabled; the others are built by uncommenting their block, and `mali_cl` needs
`-lOpenCL` added by hand.

---

## What made it fast

Nine optimisations were tried; **four survived measurement and five were
rejected**. The rejects are documented with their mechanisms in `REPORT.md`,
because a benchmark that only reports its wins is describing a different project
than the one that happened.

Kept: vectorised argmin (1.54×, it was 132M scalar iterations per frame);
blocked column sweep (2.4×, fixing a stride-15 KB access pattern);
per-core block-width tuning (~1.25×, and the optimum differs by core — A55 wants
192, A720 wants 256); non-temporal stores plus 64-byte alignment (2.8%).

Three rules the *failures* produced:

1. **A wavefront only buys parallelism when the dependency granule is smaller
   than the parallel unit.** Row-to-row dependency with row-sized work units is
   serial execution with extra atomics. This one was 2.9× *worse* and was only
   caught by measuring it.
2. **Strided writes cost more than stream count saves.** Interleaving two cost
   planes to remove a read stream was 13% worse, with L3 refills up 17% —
   falsifying the hypothesis that stream count was the binding constraint.
3. **The materialised 133 MB plane is not waste**; it is the price of
   parallelising a vertical recurrence, and every attempt to trade it away cost
   more than it saved.

---

## Traps

Things that cost real time here, recorded so they cost you less:

- 🚨 **`-t` was inert until 2026-08-28.** Every impl takes `threads` and drops it
  (`(void)threads`); the count came from `OMP_NUM_THREADS` alone, so `-t 1`
  emitted a row *labelled* single-thread holding an all-core timing. Fixed: `-t`
  now binds, and the reported count is read back from `omp_get_max_threads()`
  rather than echoed from argv. **If you have data from before that commit,
  re-run it — it cannot be repaired arithmetically**, because the error scaled
  with each machine's own thread scaling (4.63× on A55, 2.6× on A78C).
- 🚨 **`cpu_mask` in the JSON said `"0"` for every pinned run, until the same
  day.** libgomp binds the primary thread in its *constructor*, before `main`,
  so under `OMP_PROC_BIND` — which `pin.sh` sets — `sched_getaffinity()` on that
  thread reports one core however early you call it. The field now reports the
  **union over the OpenMP team**, gathered inside a parallel region. If you are
  logging "which CPUs did this run on" anywhere else, check it the same way: the
  obvious call answers a question about the asking thread, not about the work.
- 🚨 **A bit-exact kernel can still be half-scalar, and the hash will never
  tell you.** A census kernel here stepped 128 columns and stopped 124 short of
  the row end, handing 6.7% of pixels to a scalar fallback that ate ~50% of the
  phase. Output was *correct*, so correctness gating said nothing; one
  overlapping tail block took it from 92.0M to 15.8M cycles.

  **This repo ships only one of the two gates a kernel needs.** A golden file is
  a statement about outputs; it can never constrain how they were produced.
  Three things do, in increasing cost:

  1. **Absolute per-phase cost, never shares.** A share moves when any *other*
     phase moves. That census read "6.9% of runtime" and looked finished while
     hiding a 5.8×. The harness already writes absolute `census_ms`,
     `cost_ms`, `aggregate_ms`, `argmin_ms` to the JSON — the data was there
     and nobody gated on it.
  2. **An op-count roofline per phase.** Census is 62 compares over 128 lanes;
     you can compute what it *should* cost within ~2× and compare against that
     rather than against "faster than last time". Anything far off its own
     arithmetic is hiding a scalar path, a spill, or a tail.
  3. **Vector-instruction fraction.** Disassemble the hot function and count
     vector ops. A kernel that is 6.7% scalar *by pixel* is nowhere near 6.7%
     scalar by instruction, because the scalar path is a 62-iteration inner
     loop. The disassembly shows it; the timing does not.
- **gcc 12.2 cannot target `cortex-a720`.** All A720 results use
  `-mcpu=cortex-a710` (same Armv9 family, same 4×128-bit SIMD).
- **Heterogeneous clusters lie about "N cores".** The Radxa O6 has four distinct
  A720 frequency pairs, not two clusters, so a clean homogeneous 4-thread
  per-core measurement is impossible on that part. Always pin.
- **More threads is not monotonic.** The IQ-9075 peaks at 6 threads and is
  *slower* at 8; the O6 scales monotonically to 8. Same program, two machines,
  two answers — sweep, don't assume.
- **"No compute driver" is usually wrong.** Both Mali parts ship working OpenCL
  3.0; the G310 merely lacks `clinfo`, which makes it look driverless.
- **Regenerate the golden after any `sgm_params.h` change.** A stale golden with
  different `paths=` is a textbook false negative and has already happened once.

---

## Ground rules for numbers

Every figure in this repo carries one of three tags, and the distinction is
enforced rather than decorative:

- **MEASURED** — ran on real hardware, with a logged run behind it. Only these
  appear bare or in a comparison.
- **DERIVED** — computed from measurements. Always labelled, and it carries the
  conditions of *both* its factors.
- **SOURCED** — from a paper or datasheet. Always labelled.

**A DERIVED or SOURCED number is never compared against a MEASURED one.** Most
benchmark dishonesty is not fabricated data; it is a projection quietly ranked
in the same column as a measurement.

Every derived column is also a place a number can rot: it can be generated
correctly once and then go stale when its inputs move. Re-derive them
mechanically, or do not publish them. Auditing this repo's own tables that way
found a published figure wrong by 5.8×.

Power is deliberately blank everywhere. No board here exposes a usable rail for
the CPU block alone, and an empty cell beats an estimate.
