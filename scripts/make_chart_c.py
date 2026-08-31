#!/usr/bin/env python3
"""make_chart_c.py — the Configuration C results figure for the README.

Run 2 is the 1920x1080 input of the two-run pair; Run 1 is the identical
algorithm at 1920x800. ONE variable changes between them: the input size. Bit-exact to golden
0f0961d623009df5 on every row. MDE/s uses the work-performed convention
((2.074 + 0.518) Mpx x 64 / time) so both scales count.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# (label, class, ms, MDE/s work) — MEASURED. Same tiers as Configuration A:
# tuned CUDA / OpenCL / NEON+OpenMP, scalar oracle as the floor.
DATA = [
    ("NVIDIA RTX 5090 GPU",        "GPU",    11.96, 13870),
    ("NVIDIA Thor GPU",            "GPU",    56.7,   2928),
    ("NVIDIA Orin AGX GPU",        "GPU",    96.9,   1711),
    ("Cortex-A720 x8",             "CPU",   239.6,    692),
    ("Cortex-A78C x8 (6 threads)", "CPU",   291.8,    569),
    ("Hexagon v73 NSP @1.46GHz",   "DSP",   372.4,    446),
    ("Mali-G720 (10 CU)",          "GPU",   509.4,    326),
    ("Cortex-A78C x1",             "CPU",   595.3,    279),
    ("Cortex-A720 x1",             "CPU",   643.1,    258),
    ("Cortex-A55 x6 @1.8GHz",      "CPU",   844.8,    196),
    ("Cortex-A55 x1 @1.8GHz",      "CPU",  2954.0,     56),
    ("Mali-G310 (1 CU)",           "GPU",  3657.0,     45),
    ("scalar reference (1 core)",  "REF", 18020.0,    9.2),
]
COL = {"GPU": "#4C8BF5", "CPU": "#E8710A", "DSP": "#12B5A5", "REF": "#9AA0A6",
       "HW": "#7B4FD1"}

# The fixed-function engines cannot run this configuration's algorithm, so they
# have no MDE/s in the work-performed convention and never get a solid bar.
# They are placed by WALL-CLOCK EQUIVALENCE instead: the MDE/s a bit-exact
# implementation would have to reach to match that engine's frame time. Same
# quantity the tables print as "x faster (time only)". Hatched, never solid.
WORK_DE = (2.074 + 0.518) * 1e6 * 64          # this configuration's work per frame
OFA = [("NVIDIA Thor OFA*",     "HW",  4.08),
       ("NVIDIA Orin AGX OFA*", "HW", 19.51)]
DATA = DATA + [(l, c, ms, WORK_DE / (ms / 1000) / 1e6) for l, c, ms in OFA]

rows = sorted(DATA, key=lambda r: -r[3])
labels = [r[0] for r in rows]
y = range(len(rows))

fig, ax = plt.subplots(figsize=(11.5, 7.4))
for i, (lab, cls, ms, mde) in enumerate(rows):
    hw = (cls == "HW")
    ax.barh(i, mde, color=("white" if hw else COL[cls]),
            edgecolor=COL[cls], hatch=("///" if hw else None), linewidth=(1.4 if hw else 0))
    fps = 1000.0 / ms
    fps_s = f"{fps:,.0f} fps" if fps >= 10 else f"{fps:.2f} fps"
    txt = (f"({ms:,.2f} ms · {fps_s})  time-equivalent only" if hw
           else f"{mde:,.0f}   ({ms:,.1f} ms · {fps_s})")
    ax.text(mde * 1.08, i, txt, va="center", fontsize=9,
            color=(COL["HW"] if hw else "black"), fontweight=("bold" if hw else "normal"))

ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(2, 900000)
ax.set_xlabel("MDE/s, work-performed convention  (log scale)", fontsize=10)
ax.set_title("RUN 2 — 1920×1080 stereo input  ->  960×540 disparity map\n"
             "One workload at two input sizes; Run 1 is the same algorithm on 1920×800. The ONLY variable is the input size.\n"
             "D=64, 8 paths, 9×7 census on SobelX, two scales fused · solid bars bit-exact to golden 0f0961d623009df5",
             fontsize=12, pad=14)
handles = [Patch(color=COL[c], label=l) for c, l in
           [("GPU","GPU"),("CPU","CPU"),("DSP","DSP"),("REF","scalar reference")]]
handles.append(Patch(facecolor="white", edgecolor=COL["HW"], hatch="///",
                     label="fixed-function OFA — own algorithm, placed by wall-clock"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
          frameon=False, fontsize=9)
ax.grid(axis="x", alpha=.25, which="both")
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
fig.text(0.5, 0.012,
         "HATCHED BARS ARE NOT A WORK CLAIM — the OFA engines run their OWN SGM at matched in/out geometry (ds=2, D=128), are not bit-exact, "
         "and have no MDE/s here.\nEach is placed where a bit-exact implementation would have to land to MATCH ITS FRAME TIME: the same "
         "'time only' equivalence the tables print.",
         ha="center", va="bottom", fontsize=8.5, color="#7B4FD1", fontweight="bold")
fig.tight_layout()
fig.subplots_adjust(bottom=0.20)
fig.savefig("docs/run2_mde.png", dpi=150)
print("wrote docs/run2_mde.png")
