# hvx/ — Hexagon v73 (HVX) implementations

Qualcomm Hexagon NSP ports of both benchmark configurations, built on the
FastRPC project layout. Each directory holds the DSP implementation
(`*_imp.c` + kernels header), the FastRPC IDL, the aarch64 host driver, and the
skel build fragment. Build requires the Hexagon SDK (6.x, v73 target) and follows
the calculator-example project layout; the host compiles with a stock
`aarch64-linux-gnu-gcc` against the board's `libcdsprpc.so`.

| dir | config | golden | board result (IQ-9075, 4 HVX threads, median of ≥5 invocations) |
|---|---|---|---|
| `configA/` | primary (D parametric 32–256) | per scene/D | 1080p D=64: 148.44 ms |
| `configB/` | multiscale 8-path | `bcb9cb0bd6f49799` | 276.26 ms · Config C (same kernels, 1080p scene): 372.38 ms, `0f0961d623009df5` |

Board numbers quote medians across separate invocations (the board is bimodal
between invocations; see the main README's estimator notes). Correctness is gated
in-run: the hosts take `-G <fnv1a>` and print `THIS TIMING IS VOID` on mismatch.
