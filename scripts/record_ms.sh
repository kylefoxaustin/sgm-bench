#!/bin/sh
# record_ms.sh — turn a Configuration B/C run into a machine-readable record.
#
#   ./scripts/record_ms.sh <board> <run1|run2> <binary> [binary args...]
#
# WHY THIS EXISTS. The `results/*.json` records written by common/harness.c are
# Configuration A only: that harness serves the 4-path, P1=20, P2=192 build and
# stamps those parameters into every line it writes. Configurations B and C are
# standalone binaries with their own parameters, and until now their evidence
# lived only as console output pasted into REPORT.md — so a reader checking the
# repository against the published B/C tables found records that disagreed with
# them, because they were records of a different configuration.
#
# Every multiscale implementation (ref, NEON, OpenCL, CUDA, HVX) prints the same
# line, so this one recorder serves all of them and any board.
set -eu
BOARD=$1; RUN=$2; shift 2
OUT=${OUT:-results/multiscale_runs.json}
LINE=$("$@" 2>/dev/null | grep -E "^ms_" | tail -1)
[ -n "$LINE" ] || { echo "no result line from: $*" >&2; exit 1; }

echo "$LINE" | BOARD="$BOARD" RUN="$RUN" OUT="$OUT" CMD="$*" python3 -c '
import sys, re, os, json, subprocess
line = sys.stdin.read().strip()
m = re.search(r"^(\S+)\s+in (\d+)x(\d+) out (\d+)x(\d+) D=(\d+) paths=(\d+)(?: th=(\d+))?\s+"
              r"median ([\d.]+) ms\s+fps ([\d.]+)\s+MDE/s\(work\) (\d+)\s+hash (\w+)(.*)$", line)
if not m: sys.exit("unparsed: " + line)
impl,W,H,ow,oh,D,paths,th,ms,fps,mde,h,tail = m.groups()
try: sha = subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
except Exception: sha = "unknown"
rec = {
  "impl": impl, "board": os.environ["BOARD"], "config": os.environ["RUN"],
  "W": int(W), "H": int(H), "out_W": int(ow), "out_H": int(oh),
  "D": int(D), "paths": int(paths), "threads": int(th) if th else 0,
  # Configuration B/C parameters, which differ from Configuration A and are the
  # reason a Configuration A record can never stand in for one of these:
  "P1_horizontal": 40, "P1_vertical": 20, "P1_diagonal": 10, "P2": 200,
  "census": "9x7/62-bit on SobelX", "scales": 2, "scale_fusion": "cost average",
  "median_ms": float(ms), "fps": float(fps), "mde_s_work": int(mde),
  "hash": h, "golden_match": 1 if "GOLDEN OK" in tail else 0,
  "cmd": os.environ["CMD"], "git": sha,
}
with open(os.environ["OUT"], "a") as f: f.write(json.dumps(rec) + "\n")
print("recorded:", rec["board"], rec["config"], rec["median_ms"], "ms", rec["hash"],
      "GOLDEN OK" if rec["golden_match"] else "*** GOLDEN MISMATCH ***")
'
