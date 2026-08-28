# sgm-bench

**How fast can a CPU do Semi-Global Matching, if you refuse to let it be wrong?**

A from-scratch SGM stereo implementation benchmarked across seven execution
targets — x86-64, Cortex-A55, Cortex-A720, Cortex-A78C, two Mali GPUs via
OpenCL, and a Hexagon NSP via FastRPC — where **every target must produce the
same output, byte for byte**, or its timing is thrown away.

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
| 8× Cortex-A720 (Radxa O6) | CPU | 8 | 51.7 | **2,569** | 423 |
| 8× Cortex-A78C (IQ-9075) | CPU | 6 | 94.7 | 1,402 | 402 |
| 6× Cortex-A55 (i.MX 95) | CPU | 6 | 341.3 | 389 | 82 |
| Mali-G720, 10 CU | GPU | — | 256 | 518 | — |
| Hexagon v73 NSP | DSP | 6 | 1223.0 | 108 | — |
| Mali-G310, 1 CU | GPU | — | 1846 | 72 | — |
| scalar reference, 1× A55 | floor | 1 | 9188 | 14.4 | — |

**MDE/s** = million disparity estimations/sec = `W × H × D / runtime`. It is the
unit the literature uses because fps hides D — two implementations can both
report "30 fps" while doing 100× different work. It normalises D but **not**
census window size or path count, so compare those explicitly.

The headline finding is not the top row. It is that **both wide-SIMD targets
lose to the out-of-order CPUs** — the 10-CU Mali loses to *two* A720 cores, and
the Hexagon NSP loses to its own die's CPU cluster by an order of magnitude.
SGM's aggregation is a serial recurrence in x or y, so a wide unit can fill
lanes but cannot hide the dependency, while an OoO window absorbs exactly that
shape. (Caveats on both, including work still undone on the DSP, are in
`REPORT.md` — the DSP number has already improved once and may again.)

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
