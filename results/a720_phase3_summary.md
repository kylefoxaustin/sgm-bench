# Phase 3 — A720 (Radxa Orion O6, CIX CD8180), 4-path SGM, bit-exact

Board: Radxa Orion O6. Debian 12, CIX vendor kernel 6.1.44, gcc 12.2.0.
Governor set to performance for all runs.

## ⚠️ TOPOLOGY CORRECTION TO CLAUDE.md

The board table says "Cortex-A720 cluster(s) ... there are two at different
clocks". MEASURED by MIDR, there are FOUR A720 frequency pairs plus a 4-core
A520 cluster:

| cpus  | MIDR part | core | max MHz |
|-------|-----------|------|---------|
| 0, 11 | 0xd81     | A720 | 2500    |
| 9, 10 | 0xd81     | A720 | 2400    |
| 5, 6  | 0xd81     | A720 | 2300    |
| 7, 8  | 0xd81     | A720 | 2200    |
| 1-4   | 0xd80     | A520 | 1800    |

No four A720s share a clock, so a clean homogeneous 4-thread per-core number is
impossible on this part. Every run below logs its frequency set.

## ⚠️ COMPILER DEVIATION FROM THE SPEC, DOCUMENTED

The spec asks for GCC 13+ and `-mcpu=cortex-a720`. The board has gcc 12.2.0,
which does NOT know cortex-a720 or cortex-a715 (newest big core it accepts is
cortex-a710). No gcc-13 in the configured repos, no backports, clang is 14.
Rather than add repositories to a shared board for a marginal codegen
difference, all A720 numbers use **-mcpu=cortex-a710** -- the A720's immediate
predecessor, same Armv9 family and same 4x128-bit SIMD configuration.
A gcc-13+ rebuild could move these numbers and has not been done.

## Results, 1920x1080, 4-path, BW=256 (measured optimum, see below)

### D=64
| threads | cpus              | clocks     | ms    | Mpix/s | MDE/s |
|---------|-------------------|------------|-------|--------|-------|
| 1       | 0                 | 2.5        | 313.6 |  6.61  |  423  |
| 2       | 0,11              | 2.5        | 163.5 | 12.68  |  811  |
| 4       | 0,11,9,10         | 2.4-2.5    |  92.1 | 22.51  | 1441  |
| 8       | 0,5,6,7,8,9,10,11 | 2.2-2.5    |  52.6 | 39.41  | 2522  |

### D=128 (the configuration published ARM work uses)
| BW  | ms    | Mpix/s | MDE/s |
|-----|-------|--------|-------|
| 128 | 118.4 | 17.52  | 2243  |
| 192 | 136.7 | 15.17  | 1942  |
| 256 | 105.9 | 19.59  | 2507  |
| 320 | 122.4 | 16.95  | 2170  |

All GOLDEN OK. Hashes match the A55 and x86_64 oracles exactly:
  D=64  paths=4  46470bd7a464469d
  D=128 paths=4  0647dfbcf5cd0c74
(three architectures, one golden -- the oracle is portable.)

## Block width is sharply tuned and NOT the same as the A55's

A55 optimum BW=192; A720 optimum BW=256. The A720 curve is non-monotonic and
the dip is narrow and reproducible:
  224 -> 67.2ms | 240 -> 60.8 | 256 -> 52.5 | 272 -> 53.1 | 320 -> 64.8
BW=256 gives a 16 KB state array (256 x 64 B), a clean power of two.
Tuning BW per target is worth ~25%; inheriting the A55 value costs it.

## ⭐ SCANLINE INTERLEAVING: MEASURED, AND IT DOES NOT HELP

The spec predicts: "The recurrence is a ~5-op dependency chain per pixel per
direction. One scanline cannot fill four pipes. Interleave K independent
scanlines (K = 2, 3, 4 -- measure each)."

PREDICTION RECORDED BEFORE THE RUN: little or no gain, because IPC was already
2.63. RESULT (D=128, BW=256, 8 threads), K interleaved rows in pass 2:

| K | ms    | vs K=1 |
|---|-------|--------|
| 1 | 106.9 | --     |
| 2 | 107.7 | -0.7%  |
| 3 | 107.7 | -0.7%  |
| 4 | 110.3 | -3.2%  |

The premise is correct -- a single scanline's recurrence cannot fill four SIMD
pipes -- but the code already solves it somewhere else: pass 1's column-block
loop runs BW (=256) INDEPENDENT column recurrences back to back, which fills the
out-of-order window on its own. Measured IPC 2.63 on A720 vs 0.62 on the
in-order A55 for the same source.

⭐ CACHE BLOCKING AND SCANLINE INTERLEAVING ARE TWO SOLUTIONS TO ONE PROBLEM.
Having done the first for the A55's cache, the second has nothing left to buy,
and past K=2 the extra live state (K x DREGS = 32 vectors at K=4/D=128) costs
register pressure -- which is the spill the spec itself warned about, showing up
as the 3% regression.

## Cross-platform, best per-core rate observed at any thread count

| platform            | MDE/s total | best per-core MDE/s | note                |
|---------------------|-------------|---------------------|---------------------|
| 8x A720 @2.2-2.5GHz | 2522 (D=64) | 423 (1t @2.5)       | IPC 2.63            |
| 6x A55  @1.8GHz     |  375 (D=64) |  82 (1t)            | IPC 0.62, DRAM-bound|

ONE A720 core (423) beats ALL SIX A55s (375). Same source, same flags family.

## Against published work (SOURCED: ReS2tAC, Sensors 2021, Tables 6/7)

| implementation             | D   | MDE/s | hardware              |
|----------------------------|-----|-------|-----------------------|
| Zhao et al. CT5x5-SGM      | 128 | 9590  | FPGA                  |
| Rahnama et al. CT5x5-MGM   | 128 | 4247  | FPGA                  |
| **this work, 8x A720**     | 128 | **2507** | **CPU**            |
| Hernandez-Juarez CT9x7-SGM | 128 |  747  | GPU (Jetson TX1)      |
| ReS2tAC-CUDA CT9x7-SGM     | 128 |  645  | GPU (Xavier AGX)      |
| ReS2tAC-NEON 4-path CT5x5  | 128 |  242  | CPU (8x Carmel)       |
| ReS2tAC-NEON 8-path CT5x5  | 128 |  166  | CPU (8x Carmel)       |

10.4x the published ARM-CPU record at matched D and matched path count.
3.4x the best embedded-GPU SGM in that table, from a CPU.

CAVEAT, stated because it cuts against us: our census is CT9x7 (62-bit
descriptor) where theirs is CT5x5 (24-bit), so we do MORE work per pixel --
that makes the throughput comparison conservative. But the image content and
the benchmark harness differ, so this is indicative, not a controlled
head-to-head. The controlled axes (D, path count) are matched; the cost
function and the data are not.
