# Phase 2 — A55 (i.MX 95 FRDM), 4-path SGM, bit-exact

Board: NXP FRDM-IMX95-PRO, 6x Cortex-A55 (MIDR 0xd05) @ 1.8 GHz (measured),
governor=performance, no throttling observed (1800000 kHz before and after
every run). gcc 15.2.0, -O3 -mcpu=cortex-a55 -fopenmp.
Cache (measured): L1D 32K/core, L2 64K/core, L3 512K shared by all 6.

All runs GOLDEN OK against the scalar oracle. Hashes:
  D=64  paths=4  b1b407b5949f0cc1
  D=128 paths=4  0647dfbcf5cd0c74

## D=64, 1920x1080, block width 192

| threads | ms   | Mpix/s | MDE/s | scaling |
|---------|------|--------|-------|---------|
| 1       | 1615 | 1.28   |  82   | 1.00x   |
| 2       |  818 | 2.54   | 163   | 1.97x   |
| 4       |  482 | 4.31   | 276   | 3.35x   |
| 6       |  354 | 5.86   | 375   | 4.57x   |

## D=128, 1920x1080 (the configuration published ARM work uses)

| BW  | ms  | Mpix/s | MDE/s |
|-----|-----|--------|-------|
| 64  | 773 | 2.68   | 343   |
| 96  | 777 | 2.67   | 342   |
| 128 | 730 | 2.84   | 364   |
| 192 | 719 | 2.88   | 369   |

## Against published ARM-CPU SGM (SOURCED, ReS2tAC Sensors 2021 Tables 6/7)

| implementation            | D   | MDE/s | hardware              |
|---------------------------|-----|-------|-----------------------|
| this work, 4-path         | 128 | 369   | 6x A55 @1.8GHz        |
| this work, 4-path         | 64  | 375   | 6x A55 @1.8GHz        |
| ReS2tAC-NEON, 4-path CT5x5| 128 | 241.8 | 8x Carmel, Xavier AGX |
| ReS2tAC-NEON, 8-path CT5x5| 128 | 166.2 | 8x Carmel, Xavier AGX |

1.53x the published ARM record at matched D, on fewer and weaker cores, with a
2.6x more expensive census (62-bit CT9x7 vs their 24-bit CT5x5).
CAVEAT: different image content and a different census window, so this is
indicative rather than a controlled head-to-head. The controlled axis (D) is
matched; the cost function is not.

## Optimisation history — every step measured, bit-exact throughout

| step                                   | 6-thread ms | note                                    |
|----------------------------------------|-------------|-----------------------------------------|
| initial (scalar argmin, serial pass 2) | 1733        | thread scaling only 1.27x               |
| + vectorised argmin                    | 1128        | 64 scalar iters/pixel -> ~20 vector ops |
| + three parallel passes (column-major) | 1376        | scaling 3.45x but 3.3x WORSE 1-thread   |
| + blocked column sweep                 |  463        | fixed the stride-15KB access pattern    |
| + block width 192 (measured optimum)   |  354        | 256 regresses: state exceeds 64K L2     |

Total 4.9x. Step 3 is the cautionary one: the right parallel axis with the
wrong access pattern was a NET LOSS, and would have shipped as an improvement
if single-thread had not been measured separately.

## Where the remaining headroom is (measured, not guessed)

- IPC 0.62 (A55 tops near 2.0) -> not issue-bound.
- ~1.07 GB/frame of L3 refills -> DRAM-bound. Traffic is the two materialised
  planes (D and U, 133 MB each written and read) plus the per-row R buffer:
  6 threads x 123 KB exceeds the 512 KB L3.
- load_cost is evaluated 4x per pixel (D, U, R, L sweeps) where 1-2 would do.
- NOTE: CLAUDE.md assumes the W x D row buffers fit "the A55's L2". They do not
  on this part -- L2 is 64 KB, the buffers are 123 KB. That assumption should be
  corrected in the spec.

## Cheap-fix batch (non-temporal stores, prefetch) — 2x2 factorial, MEASURED

Predicted before running: "stnp worth 10-20% on the DRAM-bound A55; prefetch
fixes the load-use stalls behind IPC 0.62". BOTH PREDICTIONS WERE WRONG.

A55, 6 threads, BW=192, n=10 frames, within-arm spread 0.09 ms:

|        | NT=0    | NT=1    |
|--------|---------|---------|
| PF=0   | 353.71  | 347.99  |
| PF=1   | 356.11  | 346.36  |

  non-temporal stores : -1.6%  (real, far below the predicted 10-20%)
  prefetch alone      : +0.7%  WORSE
  A720, same change   : -0.8%  (less memory-bound, smaller win -- consistent)

DEFAULTS SET TO WHAT MEASURED BEST: SGM_NT=1, SGM_PF=0.

Why the prefetch reasoning failed: "IPC 0.62 means load-use stalls, so prefetch
the descriptors" identified the right symptom and the wrong stream. The
descriptor reads are already sequential and handled by the hardware prefetcher;
the stalls are elsewhere (most likely the plane reads in pass 2). Prefetching a
stream that was never the problem only adds instructions.

## Best numbers after the batch

| platform            | ms     | Mpix/s | MDE/s |
|---------------------|--------|--------|-------|
| 6x A55 @1.8GHz      | 346.36 |  5.99  |  383  |
| 8x A720 @2.2-2.5GHz |  51.66 | 40.14  | 2569  |

## Cache / DDR-latency techniques — the complete set, all measured

A55, 6 threads, BW=192, n=10, three repeats each (within-arm spread ~0.1 ms).

| technique                        | effect  | verdict                              |
|----------------------------------|---------|--------------------------------------|
| non-temporal stores (stnp)       | -1.6%   | KEPT                                 |
| 64-byte alignment of all buffers | -1.2%   | KEPT (malloc gave page+16)           |
| transparent huge pages           | n/a     | already [always], 157 MB live        |
| software prefetch                | +0.7%   | REJECTED -- worse                    |
| interleaved [D|U] plane          | +13%    | REJECTED -- much worse               |

Cumulative kept: 353.71 -> 343.7 ms (2.8%). A55 = 6.04 Mpix/s = 387 MDE/s.

### ⭐ THE INTERLEAVED-PLANE RESULT FALSIFIED MY MODEL, AND THAT IS THE FINDING

HYPOTHESIS (mine): the A55 is limited by STREAM COUNT -- roughly 6 concurrent
sequential streams per thread x 6 threads, against a prefetcher that tracks
8-16. Predicted: merging the D and U planes into one interleaved [D|U] buffer
removes a read stream from pass 2 and should therefore help.

RESULT: 13% WORSE (343 -> 389 ms). MECHANISM CONFIRMED, not inferred:
    l3d_cache_refill  6,834,838 (separate)  ->  7,989,592 (interleaved)  +17%
The +17% in DRAM refills tracks the +13% in time.

WHY: interleaving makes pass 1's writes STRIDED -- 64 bytes written, 64 skipped.
That defeats full-cache-line write bursts, and the memory controller can no
longer coalesce. The cost of losing sequential full-line writes is far larger
than the saving from merging two read streams into one.

⇒ WRITES ARE THE SENSITIVE SIDE ON THIS PART, NOT READ-STREAM COUNT.

CONSEQUENCE FOR THE PLANNED RESTRUCTURE: my ~2x estimate for the fused
pipelined-band rewrite rested on two arguments -- (a) traffic halves because the
U plane disappears, and (b) stream count drops. Argument (b) is now DISPROVEN.
Argument (a) still stands, and killing the U plane REMOVES a write stream rather
than making one strided, so it should still help -- but the estimate should be
revised DOWN and treated as untested. Predicting 2x on the strength of a model
that has just been falsified would be exactly the mistake this session keeps
catching.

## ⭐ THE STRUCTURAL RESTRUCTURE — ATTEMPTED, MEASURED, REVERTED

PREDICTION (recorded before the run): 10-25% on A55. KILL CRITERION: <5% -> revert.
RESULT: 1006 ms vs 343.7 ms baseline -- 2.9x WORSE. Bit-exact (GOLDEN OK), and
reverted per the criterion.

The plan was to fuse the path pairs (L+U in one forward traversal, R+D in one
backward traversal, halving cost evaluations) and to delete the 133 MB U plane
by making the vertical recurrence parallel with a per-row WAVEFRONT instead of
a materialised plane.

TWO ERRORS, one caught before measuring and one only by measuring:

1. CAUGHT BEFORE RUNNING: the first draft used contiguous row BANDS. Band b+1
   needs the U state of band b's LAST row, so it cannot begin until band b has
   finished -- serialisation dressed as a pipeline. Rewritten as a per-row
   wavefront.

2. CAUGHT ONLY BY MEASURING: the per-row wavefront gives NO PARALLELISM EITHER.
   Row y's L+U pass cannot start until row y-1's has finished, and the unit of
   work IS a whole row, so the L+U phase runs strictly serially across all 1080
   rows. (Worse, the wait was placed before the independent R sub-pass, so even
   that could not overlap.)

⭐ THE RULE: A WAVEFRONT ONLY BUYS PARALLELISM WHEN THE DEPENDENCY GRANULE IS
SMALLER THAN THE PARALLEL UNIT. Here the dependency is row-to-row and the unit
is a row, so the pipeline has depth 1 -- which is just serial execution with
extra atomics.

⭐ AND THE POSITIVE RESULT HIDING INSIDE THE NEGATIVE ONE: the U path has NO
column dependency, so it parallelises perfectly over columns. The original
design was already correct, and the 133 MB plane is not waste -- IT IS THE PRICE
OF PARALLELISING A VERTICAL RECURRENCE. Trading it away costs more than it saves.

Every remaining idea for removing that plane runs into the same wall, so the
plane stays. A55 stands at 343.7 ms / 6.04 Mpix/s / 387 MDE/s.
