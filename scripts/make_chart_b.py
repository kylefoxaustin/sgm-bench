#!/usr/bin/env python3
"""make_chart_b.py — the Configuration B results figure for the README.

Every bar is MEASURED at Configuration B: 1920x800 in, 960x400 out, two full
64-disparity scales with averaged data costs, census 9x7 on SobelX, eight paths
with direction-weighted P1, P2=200 saturating. Bit-exact to golden
bcb9cb0bd6f49799 on every row. MDE/s uses the work-performed convention
((0.384 + 1.536) Mpx x 64 / time) so both scales count.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# (label, class, ms, MDE/s work) — MEASURED. Same tiers as Configuration A:
# tuned CUDA / OpenCL / NEON+OpenMP, scalar oracle as the floor.
DATA = [
    ("NVIDIA RTX 5090 GPU",        "GPU",     9.08, 13538),
    ("NVIDIA Thor GPU",            "GPU",    45.1,   2723),
    ("NVIDIA Orin AGX GPU",        "GPU",    72.8,   1688),
    ("Cortex-A78C x8 (6 threads)", "CPU",   211.9,    580),
    ("Cortex-A720 x8",             "CPU",   238.3,    516),
    ("Hexagon v73 NSP @1.46GHz",   "DSP",   276.3,    445),
    ("Mali-G720 (10 CU)",          "GPU",   380.0,    323),
    ("Cortex-A78C x1",             "CPU",   442.6,    278),
    ("Cortex-A720 x1",             "CPU",   470.9,    261),
    ("Cortex-A55 x6 @1.8GHz",      "CPU",   640.1,    192),
    ("Cortex-A55 x1 @1.8GHz",      "CPU",  2201.0,     56),
    ("Mali-G310 (1 CU)",           "GPU",  2704.0,     45),
    ("scalar reference (1 core)",  "REF", 14265.0,    8.6),
]
COL = {"GPU": "#4C8BF5", "CPU": "#E8710A", "DSP": "#12B5A5", "REF": "#9AA0A6"}

rows = sorted(DATA, key=lambda r: -r[3])
labels = [r[0] for r in rows]
y = range(len(rows))

fig, ax = plt.subplots(figsize=(11.5, 7.4))
for i, (lab, cls, ms, mde) in enumerate(rows):
    ax.barh(i, mde, color=COL[cls])
    fps = 1000.0 / ms
    fps_s = f"{fps:,.0f} fps" if fps >= 10 else f"{fps:.2f} fps"
    ax.text(mde * 1.08, i, f"{mde:,.0f}   ({ms:,.1f} ms · {fps_s})", va="center", fontsize=9)

ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(2, 80000)
ax.set_xlabel("MDE/s, work-performed convention  (log scale)", fontsize=10)
ax.set_title("Configuration B: 1920×800, D=64, 8 paths, 9×7 census on SobelX, two scales fused, half-res output\n"
             "solid bars bit-exact to golden bcb9cb0bd6f49799 · MDE/s is the work-performed\n"
             "convention, both full-search scales count",
             fontsize=12, pad=14)
ax.legend(handles=[Patch(color=COL[c], label=l) for c, l in
                   [("GPU","GPU"),("CPU","CPU"),("DSP","DSP"),("REF","scalar reference")]],
          loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4,
          frameon=False, fontsize=9)
ax.grid(axis="x", alpha=.25, which="both")
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.subplots_adjust(bottom=0.14)
fig.savefig("docs/results_b.png", dpi=150)
print("wrote docs/results_b.png")
