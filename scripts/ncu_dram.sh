#!/bin/sh
# ncu_dram.sh — MEASURE DRAM bytes per frame on an NVIDIA part, all three configs.
#
# Needs root: NVIDIA restricts performance counters to admin unless the driver
# is built with NVreg_RestrictProfilingToAdminUsers=0.
#
#   sudo ./scripts/ncu_dram.sh
#
# On GB202 the counter names take an _op_read / _op_write suffix; the more
# commonly cited dram__bytes_read.sum does not exist there and returns n/a.
# Bytes are summed over every kernel of ONE frame. The frame time comes from a
# separate unprofiled run, because profiling serialises and replays kernels and
# its wall clock is meaningless.
set -eu
NCU=${NCU:-/usr/local/cuda/bin/ncu}
M=dram__bytes_op_read.sum,dram__bytes_op_write.sum,dram__bytes.sum
OUT=${OUT:-/tmp/ncu_dram}
mkdir -p "$OUT"

run() {                       # run <tag> <binary> <args...>
  tag=$1; shift
  echo "== $tag: bytes"
  "$NCU" --metrics "$M" --csv "$@" > "$OUT/$tag.csv" 2>"$OUT/$tag.err" || true
  echo "== $tag: time (unprofiled)"
  "$@" > "$OUT/$tag.time" 2>&1 || true
}

cd "$(dirname "$0")/.."
run configA ./bin/sgm_cuda_opt data/synthetic/left.pgm data/synthetic/right.pgm \
    -g data/golden/synthetic.pgm -w 0 -n 1
run run1    ./bin/sgm_ms_cuda  data/multiscale/left.pgm data/multiscale/right.pgm \
    -g data/multiscale/golden_ms.pgm -n 1
run run2    ./bin/sgm_ms_cuda  data/synthetic/left.pgm data/synthetic/right.pgm \
    -g data/multiscale/golden_ms_1080.pgm -n 1

chown -R "${SUDO_USER:-$USER}" "$OUT"
echo "--- wrote $OUT ---"; ls -la "$OUT"
