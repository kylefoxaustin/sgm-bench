#!/usr/bin/env python3
"""geometry.py — turn a (resolution x D) timing grid into a range/FOV answer.

FOV is not a measured axis and must not be one. It does not change the kernel;
it changes the D a given minimum range REQUIRES:

    d = f*B/Z        f = (W/2)/tan(HFOV/2)

So required D scales with image WIDTH and inversely with FOV, and since work is
W*H*D with D itself proportional to W, SGM work goes as W^2*H -- CUBIC in linear
resolution, not quadratic. Halving resolution at fixed optics is ~3.4x less work.

The useful consequence, which a throughput number hides completely: a WIDER
field of view is CHEAPER. Going 90 -> 120 degrees cuts focal length 1.73x, cuts
the D needed for the same near range by 1.73x, and sees more of the scene.
"""
import json, math, sys

def f_px(W, hfov): return (W / 2) / math.tan(math.radians(hfov) / 2)
def need_D(W, hfov, B, zmin): return f_px(W, hfov) * B / zmin
def zmin_of(W, hfov, B, D):   return f_px(W, hfov) * B / D

grid = {}
skipped = 0
for line in open(sys.argv[1] if len(sys.argv) > 1 else "results/sweep_thor.json"):
    d = json.loads(line)
    # Only golden-verified rows may enter the grid. Without this filter a cell
    # that produced the WRONG ANSWER contributed its timing to the published
    # table and to every DERIVED range/FOV figure computed from it.
    if d.get("golden_match") != 1:
        skipped += 1
        continue
    grid[(d["W"], d["H"], d["D"])] = d["median_ms"]
if skipped:
    print("!! %d row(s) skipped: not golden-verified\n" % skipped, file=sys.stderr)
if not grid:
    print("no golden-verified rows; nothing to report", file=sys.stderr); sys.exit(1)

print("=== MEASURED grid: MDE/s = W*H*D/ms ===")
print("%-12s" % "res", end="")
Ds = sorted({k[2] for k in grid})
print("".join("   D=%-14d" % D for D in Ds))
res = sorted({(k[0], k[1]) for k in grid}, key=lambda r: -r[0] * r[1])
for W, H in res:
    print("%-12s" % ("%dx%d" % (W, H)), end="")
    for D in Ds:
        ms = grid.get((W, H, D))
        print("   %6.2f ms %6.0f" % (ms, W * H * D / ms / 1000) if ms else "   %-17s" % "-", end="")
    print()

print("\n=== DERIVED: what each cell buys you, at B=12 cm ===")
B = 0.12
print("%-12s %-6s %-7s %-9s %-8s %s" % ("res", "HFOV", "D used", "min range", "fps", "note"))
for W, H in res:
    for hfov in (90, 120):
        for D in Ds:
            ms = grid.get((W, H, D))
            if not ms: continue
            z = zmin_of(W, hfov, B, D)
            print("%-12s %-6d %-7d %-9.2f %-8.1f %s" % (
                "%dx%d" % (W, H), hfov, D, z, 1000 / ms,
                "real-time" if 1000 / ms >= 30 else ""))
    print()
