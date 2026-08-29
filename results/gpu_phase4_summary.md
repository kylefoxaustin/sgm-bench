# Phase 4 — Mali GPUs via OpenCL, bit-exact

One kernel source (mali_cl/sgm.cl) run unchanged on both GPUs.

## ⚠️ SPEC CORRECTION: "no driver" was NOT the answer

CLAUDE.md budgets up to a day for the G310 driver situation and says to
"record 'no driver available in BSP <version>' as the result if that's the
answer". It is not. BOTH GPUs have a working OpenCL 3.0 stack out of the box:

| board   | device                    | CUs | subgroup | driver | ICD                     |
|---------|---------------------------|-----|----------|--------|-------------------------|
| i.MX 95 | Mali-G310 r0p0            |  1  | 16       | r54p1  | /etc/OpenCL/vendors/mali-imx.icd |
| O6      | Mali-G720-Immortalis r0p0 | 10  | 16       | r53p0  | ARM Platform            |

Both expose cl_khr_subgroups, subgroup_shuffle and subgroup_clustered_reduce.
Timebox unspent.

⚠️ WHAT MAKES THE G310 *LOOK* DRIVERLESS: `clinfo` is not installed on the
i.MX 95 image. A dlopen() probe against libOpenCL.so.1 found the stack in about
two minutes. Anyone concluding "no driver" from a missing tool is wrong -- this
belongs on the board's asset card.

⚠️ AND A SELF-CORRECTION: my first device query reported "warp = 1" for the
G310, which would have invalidated the spec's whole kernel design (16 work-items
x uchar4). That was a WRONG ENUM on my part (0x1035 is not
PREFERRED_WORK_GROUP_SIZE_MULTIPLE). Asking the KERNEL rather than the device --
clGetKernelWorkGroupInfo / clGetKernelSubGroupInfo -- returned 16, which is what
Valhall should be. Verified before shaping the kernel around it.

## Results, 1920x1080, D=64, 4 paths

| device                    | CUs | ms    | Mpix/s | MDE/s | golden |
|---------------------------|-----|-------|--------|-------|--------|
| Mali-G310 (i.MX 95)       |  1  | 1846  |  1.12  |   72  | OK     |
| Mali-G720-Immortalis (O6) | 10  |  256  |  8.09  |  518  | OK     |

1 -> 10 CUs scales 7.2x, so the kernel parallelises sensibly.
Both hashes 46470bd7a464469d -- identical to the CPU oracle.

## ⭐ THE FINDING: BOTH GPUs LOSE TO THE CPU, AND THE G310 LOSES BADLY

| implementation        | MDE/s |
|-----------------------|-------|
| 8x A720 (CPU)         | 2569  |
| 4x A720 (CPU)         | 1441  |
| **G720 GPU (10 CUs)** |  518  |
| 1x A720 core (CPU)    |  423  |
| 6x A55 (CPU)          |  389  |
| 1x A55 core (CPU)     |   82  |
| **G310 GPU (1 CU)**   |   72  |

The spec predicted "it loses to 4x A720". It loses to TWO. And the G310 --
the i.MX 95's own GPU -- is beaten by a SINGLE A55 CORE, before counting the
copy in and out.

⇒ PRACTICAL CONCLUSION FOR THE i.MX 95: offloading SGM to the G310 is strictly
worse than leaving it on one A55 core. Measured, not assumed.

⇒ AND THE STRUCTURAL REASON, which also predicts the DSP case: SGM's recurrence
serialises along each sweep direction. A GPU can only parallelise across
rows/columns and across disparities -- exactly what 8 wide out-of-order cores
with 128-bit SIMD already do well, and the GPU pays latency for it. SGM suits
CPUs.

## FAIRNESS CAVEAT, stated because it cuts against the GPUs

This is a FIRST-CUT kernel: no local-memory tiling of descriptors, scalar
argmin, clFinish between every sweep, and a 265 MB global S buffer. A tuned
version would do better. But the gap is 5x on the G720 and 6x per-core on the
G310, and the structural argument above does not move with tuning.
