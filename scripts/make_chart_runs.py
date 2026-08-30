#!/usr/bin/env python3
"""make_chart_runs.py — the two-resolution figures, in FPS.

One workload (two full 64-D scales fused, 8 paths, weighted P1, census 9x7 on
SobelX, half-res output), two input sizes. Bars are fps — the axis a systems
reader actually wants — log scale so the scalar floor stays visible. Purple
bars are the fixed-function OFA engines running their OWN algorithm at matched
geometry (not bit-exact, labelled). Also dumps axis geometry to JSON so a
downstream deck can overlay markers at exact fps positions."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RUNS = {
 "run1080": dict(title="Run 1 — 1920×1080 stereo input  →  960×540 disparity map",
   golden="0f0961d623009df5", png="docs/run1080_fps.png", rows=[
    ("NVIDIA Thor OFA*",           "HW",     4.08),
    ("NVIDIA RTX 5090",            "GPU",   11.96),
    ("NVIDIA Orin AGX OFA*",       "HW",    19.51),
    ("NVIDIA Thor GPU",            "GPU",   56.7),
    ("NVIDIA Orin AGX GPU",        "GPU",   96.9),
    ("Cortex-A720 x8",             "CPU",  239.6),
    ("Cortex-A78C x8 (6 thr)",     "CPU",  291.8),
    ("Hexagon v73 NSP",            "DSP",  372.4),
    ("Mali-G720 (10 CU)",          "GPU",  509.4),
    ("Cortex-A78C x1",             "CPU",  595.3),
    ("Cortex-A720 x1",             "CPU",  643.1),
    ("Cortex-A55 x6",              "CPU",  844.8),
    ("Cortex-A55 x1",              "CPU", 2954.0),
    ("Mali-G310 (1 CU)",           "GPU", 3657.0),
    ("scalar reference (1 core)",  "REF",18020.0)]),
 "run800": dict(title="Run 2 — 1920×800 stereo input  →  960×400 disparity map",
   golden="bcb9cb0bd6f49799", png="docs/run800_fps.png", rows=[
    ("NVIDIA Thor OFA*",           "HW",     3.38),
    ("NVIDIA RTX 5090",            "GPU",    9.08),
    ("NVIDIA Orin AGX OFA*",       "HW",    14.90),
    ("NVIDIA Thor GPU",            "GPU",   45.1),
    ("NVIDIA Orin AGX GPU",        "GPU",   72.8),
    ("Cortex-A78C x8 (6 thr)",     "CPU",  211.9),
    ("Cortex-A720 x8",             "CPU",  238.3),
    ("Hexagon v73 NSP",            "DSP",  276.3),
    ("Mali-G720 (10 CU)",          "GPU",  380.0),
    ("Cortex-A78C x1",             "CPU",  442.6),
    ("Cortex-A720 x1",             "CPU",  470.9),
    ("Cortex-A55 x6",              "CPU",  640.1),
    ("Cortex-A55 x1",              "CPU", 2201.0),
    ("Mali-G310 (1 CU)",           "GPU", 2704.0),
    ("scalar reference (1 core)",  "REF",14265.0)]),
}
COL = {"GPU": "#4C8BF5", "CPU": "#E8710A", "DSP": "#12B5A5", "HW": "#7B4FD1", "REF": "#9AA0A6"}
XLIM = (0.04, 500)
geom = {}

for key, run in RUNS.items():
    rows = run["rows"]
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    for i, (lab, cls, ms) in enumerate(rows):
        fps = 1000.0 / ms
        ax.barh(i, fps, color=COL[cls])
        fps_s = f"{fps:,.0f} fps" if fps >= 10 else f"{fps:.2f} fps"
        ms_s  = f"{ms:,.0f} ms" if ms >= 100 else f"{ms:.2f} ms"
        ax.text(fps * 1.1, i, f"{fps_s}   ({ms_s})", va="center", fontsize=10)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(*XLIM)
    ax.set_xlabel("frames per second  (log scale)", fontsize=11)
    ax.set_title(run["title"] + "\n"
                 "one workload: two full 64-disparity scales fused, 8 paths, weighted P1, census 9×7 on SobelX\n"
                 f"every solid bar byte-identical to golden {run['golden']} · *purple = fixed-function OFA, its own algorithm",
                 fontsize=12, pad=14)
    ax.legend(handles=[Patch(color=COL[c], label=l) for c, l in
                       [("GPU","GPU"),("CPU","CPU"),("DSP","DSP"),
                        ("HW","fixed-function OFA (own algorithm, not bit-exact)"),("REF","scalar reference")]],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=5, frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=.25, which="both")
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.15)
    fig.savefig(run["png"], dpi=150)
    bb = ax.get_position()
    geom[key] = dict(x0=bb.x0, x1=bb.x1, y0=bb.y0, y1=bb.y1, xmin=XLIM[0], xmax=XLIM[1])
    print("wrote", run["png"])

json.dump(geom, open("docs/run_chart_geom.json", "w"), indent=1)
print("wrote docs/run_chart_geom.json")
