#!/usr/bin/env bash
# sweep.sh — the (resolution x disparity) measurement grid.
#
# Each cell is a SEPARATE BUILD, because D is compile-time, and a SEPARATE
# GOLDEN, because changing D changes the correct answer. Regenerating the golden
# per cell is the whole point: a stale golden from another D is a textbook false
# negative and has already happened once in this project.
#
#   ./scripts/sweep.sh <impl-build-cmd> <out.json>
#
# Field of view is NOT a third axis here. FOV does not change the kernel; it
# changes the D a given range REQUIRES, via d = f*B/Z with f = (W/2)/tan(HFOV/2).
# So FOV is analysed over this grid afterwards by scripts/geometry.py, not
# measured. Measuring it would be measuring the same cells twice under new names.
set -u
OUT=${2:-results/sweep.json}
NVARCH=${NVARCH:-native}
mkdir -p results data/sweep
: > "$OUT"

for D in 32 64 128; do
  for RES in "1920 1080" "1280 720" "960 540" "640 480"; do
    set -- $RES; W=$1; H=$2
    echo "=== D=$D  ${W}x${H} ==="
    gcc -O2 -Icommon -DSGM_D=$D -o bin/gen_synthetic common/gen_synthetic.c 2>/dev/null  # scene must match the D under test
    ./bin/gen_synthetic "$W" "$H" data/sweep 1 >/dev/null 2>&1 || { echo "  gen failed"; continue; }

    # golden: scalar reference, this D, this resolution
    gcc -O2 -Wall -std=gnu11 -Icommon -DSGM_D=$D -DCFLAGS_STR='"ref"' -DGIT_SHA='"sweep"' \
        -o bin/sgm_ref_d common/sgm_ref.c common/harness.c common/util.c 2>/dev/null \
        || { echo "  ref build failed"; continue; }
    ./bin/sgm_ref_d data/sweep/left.pgm data/sweep/right.pgm -w 0 -n 1 \
        -o data/sweep/golden.pgm --no-roofline >/dev/null 2>&1 \
        || { echo "  golden failed"; continue; }

    # implementation under test, same D
    if ! eval "${1:-false}" ; then :; fi
    nvcc -O3 -Icommon -arch=$NVARCH -DSGM_D=$D -c cuda/sgm_cuda_opt.cu -o /tmp/sw.o 2>/dev/null \
        && gcc -O2 -Wall -std=gnu11 -Icommon -DSGM_D=$D -DCFLAGS_STR='"nvcc -O3"' \
               -DGIT_SHA='"sweep"' -c common/harness.c -o /tmp/swh.o 2>/dev/null \
        && gcc -O2 -Icommon -c common/util.c -o /tmp/swu.o 2>/dev/null \
        && nvcc -o bin/sgm_sweep /tmp/swh.o /tmp/swu.o /tmp/sw.o 2>/dev/null \
        || { echo "  impl build failed"; continue; }

    # A failed cell must STOP the sweep, not scroll past it. This used to end in
    # `|| true`, which swallowed exit 2 (GOLDEN MISMATCH) so a wrong-answer row
    # landed in the output JSON and flowed on into the published grid.
    ./bin/sgm_sweep data/sweep/left.pgm data/sweep/right.pgm -g data/sweep/golden.pgm \
        -w 3 -n 12 -j "$OUT" --board "${BOARD:-unknown}" --no-roofline 2>&1 | grep -E "^cuda|GOLDEN"
    rc=${PIPESTATUS[0]}
    if [ "$rc" != 0 ]; then echo "  ABORT: cell D=$D ${W}x${H} exited $rc"; exit "$rc"; fi
  done
done
echo "grid written to $OUT"
