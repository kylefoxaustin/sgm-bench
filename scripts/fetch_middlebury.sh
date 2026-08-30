#!/usr/bin/env bash
# fetch_middlebury.sh — pull real stereo scenes. NOT committed: they are
# Middlebury's, several MB each, and redistributing them is not ours to do.
#
# Middlebury 2014 pairs are captured with a calibrated two-camera rig; the dense
# ground truth comes from structured lighting, not simulation. That is what makes
# them a real test rather than another synthetic one.
#
#   ./scripts/fetch_middlebury.sh Motorcycle  data/real
#   ./scripts/prep_real.py        data/real/Motorcycle-perfect data/real 3
set -eu
S=${1:-Motorcycle}; OUT=${2:-data/real}
B=https://vision.middlebury.edu/stereo/data/scenes2014/datasets/$S-perfect
mkdir -p "$OUT/$S-perfect"
for f in im0.png im1.png disp0.pfm calib.txt; do
  echo "  $S/$f"; curl -fsS -m 300 -o "$OUT/$S-perfect/$f" "$B/$f"
done
grep -E 'ndisp|width|height' "$OUT/$S-perfect/calib.txt"
