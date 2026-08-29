# nvofa — the discrete NVOFA stereo/optical-flow engine

`ofa_bench.c` drives the RTX 5090's fixed-function Optical Flow Accelerator
through the public [NVIDIA Optical Flow SDK](https://github.com/NVIDIA/NVIDIAOpticalFlowSDK)
headers (no login required — fetch `nvOpticalFlowCommon.h` and
`nvOpticalFlowCuda.h` from that repo).

    gcc -O2 -I. -I/usr/local/cuda/include -o ofa_bench ofa_bench.c \
        -L/usr/local/cuda/lib64 -lcuda -ldl -lm
    OF_MODE=of ./ofa_bench left.pgm right.pgm 128 60

**These numbers are NOT bit-exact to this repo's golden** and are not gated by
it: NVOFA is NVIDIA's own implementation. Throughput comparison only.
