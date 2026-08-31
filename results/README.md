# What the records in this directory do and do not cover

Read this before checking a published table against a file here, because two
different configurations live in this repository and their records look
nothing alike.

## The gap this file exists to close

Until 2026-08-31 every machine-readable record here was **Configuration A**:
`"paths":4, "P1":20, "P2":192`, written by `common/harness.c`, which serves the
4-path build and stamps those parameters into every line. Configurations **B**
and **C** are standalone binaries with different parameters — 8 paths,
direction-weighted P1 of 40/20/10, P2=200, two fused scales — and their
evidence existed only as console output quoted in `REPORT.md`.

The effect was that **a reader checking the B/C tables against this directory
found records that contradicted them**, because they were records of a
different configuration: a 4-path Configuration A number where an 8-path
Configuration B number was expected, with no file saying so. The measurements
were never wrong; the repository simply failed to substantiate its own tables.
That is exactly the failure this project's own rules are written to prevent,
and it was caught by an outside reader rather than by us.

## What is here now

| file | configuration | parameters |
|---|---|---|
| `rtx5090_runs.json`, `sweep_thor*.json` | **A** — 1920×1080, D sweep | 4 paths, P1=20, P2=192 |
| `multiscale_runs.json` | **B (run 1)** and **C (run 2)** | 8 paths, P1 40/20/10, P2=200, 2 scales |

Every record in `multiscale_runs.json` carries its own parameters explicitly —
`paths`, `P1_horizontal`, `P1_vertical`, `P1_diagonal`, `P2`, `scales`,
`census` — so no record can ever again be mistaken for one of a different
configuration.

## Re-measured against the published tables

Every row below was re-run on 2026-08-31, on the same silicon, and checked
against the published golden **in the same invocation that was timed**:

| board | run | published ms | re-measured ms | ratio | golden | gate |
|---|---|---|---|---|---|---|
| rtx5090 | run 1 | 9.08 | 9.10 | 1.002× | `bcb9cb0bd6f49799` | OK |
| rtx5090 | run 2 | 11.96 | 11.84 | 0.990× | `0f0961d623009df5` | OK |
| iq9-a78c | run 1 | 211.90 | 221.25 | 1.044× | `bcb9cb0bd6f49799` | OK |
| iq9-a78c | run 2 | 291.80 | 290.14 | 0.994× | `0f0961d623009df5` | OK |
| imx95-a55 | run 1 | 640.10 | 644.57 | 1.007× | `bcb9cb0bd6f49799` | OK |
| imx95-a55 | run 2 | 844.80 | 845.81 | 1.001× | `0f0961d623009df5` | OK |

Five of six land within 1%. The A78C's run-1 cell is 4.4% high, which is inside
the ~9% invocation-to-invocation bimodality already documented for that board
in `REPORT.md` — and is why its published figures are medians rather than
minima.

## Rows that still have no record here

`Thor`, `Orin AGX`, `Cortex-A720`, `Mali-G720`, `Mali-G310` and the
`Hexagon NSP`. Their B/C evidence is the console output in `REPORT.md`; the
boards were not reachable when this directory was reconciled. **Their numbers
are not in doubt and are not withdrawn** — but they are, today, less well
evidenced than the six rows above, and that distinction is stated here rather
than left for a reader to find.

To add one, run it on the board and commit the line:

```sh
./scripts/record_ms.sh <board> <run1|run2> <binary> <left> <right> -g <golden> -n 5
```

The recorder parses the line every multiscale implementation prints — ref,
NEON, OpenCL, CUDA and HVX alike — so one command serves any target.
