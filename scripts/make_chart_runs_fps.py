#!/usr/bin/env python3
"""make_chart_runs_fps.py — the two-run figures, frame-rate first.

One workload at two input sizes. X axis is FRAMES PER SECOND on a LINEAR scale;
each bar is labelled with its fps and, in parentheses, the MDE/s work rate and
the wall-clock milliseconds behind it. Solid bars are bit-exact to that run's
golden; hatched bars are the fixed-function OFA engines running their OWN
algorithm at matched geometry, placed by wall-clock equivalence and carrying no
MDE/s.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COL = {"GPU": "#4C8BF5", "CPU": "#E8710A", "DSP": "#12B5A5", "REF": "#9AA0A6", "HW": "#7B4FD1"}

# label, class, ms, MDE/s (None = fixed-function engine, own algorithm)
RUN1 = [  # 1920x800 in -> 960x400 out, golden bcb9cb0bd6f49799
    ("NVIDIA Thor OFA*",          "HW",     3.38, None),
    ("NVIDIA RTX 5090 GPU",       "GPU",    9.08, 13538),
    ("NVIDIA Orin AGX OFA*",      "HW",    14.90, None),
    ("NVIDIA Thor GPU",           "GPU",   45.1,   2723),
    ("NVIDIA Orin AGX GPU",       "GPU",   72.8,   1688),
    ("Cortex-A78C x8 (6 threads)","CPU",  211.9,    580),
    ("Cortex-A720 x8",            "CPU",  238.3,    516),
    ("Hexagon v73 NSP",           "DSP",  276.3,    445),
    ("Mali-G720 (10 CU)",         "GPU",  380.0,    323),
    ("Cortex-A78C x1",            "CPU",  442.6,    278),
    ("Cortex-A720 x1",            "CPU",  470.9,    261),
    ("Cortex-A55 x6",             "CPU",  640.1,    192),
    ("Cortex-A55 x1",             "CPU", 2201.0,     56),
    ("Mali-G310 (1 CU)",          "GPU", 2704.0,     45),
    ("scalar reference (1 core)", "REF",14265.0,    8.6),
]
RUN2 = [  # 1920x1080 in -> 960x540 out, golden 0f0961d623009df5
    ("NVIDIA Thor OFA*",          "HW",     4.08, None),
    ("NVIDIA RTX 5090 GPU",       "GPU",   11.96, 13870),
    ("NVIDIA Orin AGX OFA*",      "HW",    19.51, None),
    ("NVIDIA Thor GPU",           "GPU",   56.7,   2928),
    ("NVIDIA Orin AGX GPU",       "GPU",   96.9,   1711),
    ("Cortex-A720 x8",            "CPU",  239.6,    692),
    ("Cortex-A78C x8 (6 threads)","CPU",  291.8,    569),
    ("Hexagon v73 NSP",           "DSP",  372.4,    446),
    ("Mali-G720 (10 CU)",         "GPU",  509.4,    326),
    ("Cortex-A78C x1",            "CPU",  595.3,    279),
    ("Cortex-A720 x1",            "CPU",  643.1,    258),
    ("Cortex-A55 x6",             "CPU",  844.8,    196),
    ("Cortex-A55 x1",             "CPU", 2954.0,     56),
    ("Mali-G310 (1 CU)",          "GPU", 3657.0,     45),
    ("scalar reference (1 core)", "REF",18020.0,    9.2),
]

RUNS = [
    ("run1_fps.png", "RUN 1 — 1920×800 stereo input  ->  960×400 disparity map",
     "bcb9cb0bd6f49799", "Run 2 is the same algorithm on 1920×1080 — the ONLY variable is the input size.", RUN1),
    ("run2_fps.png", "RUN 2 — 1920×1080 stereo input  ->  960×540 disparity map",
     "0f0961d623009df5", "Run 1 is the same algorithm on 1920×800 — the ONLY variable is the input size.", RUN2),
]

for fname, title, golden, sub, data in RUNS:
    rows = sorted(data, key=lambda r: -(1000.0 / r[2]))
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    xmax = 1000.0 / rows[0][2]
    for i, (lab, cls, ms, mde) in enumerate(rows):
        fps = 1000.0 / ms
        hw = (cls == "HW")
        ax.barh(i, fps, height=0.68,
                color=("white" if hw else COL[cls]), edgecolor=COL[cls],
                hatch=("///" if hw else None), linewidth=(1.4 if hw else 0))
        fps_s = f"{fps:,.1f}" if fps < 100 else f"{fps:,.0f}"
        if hw:
            txt = f"{fps_s} fps   ({ms:,.2f} ms · own algorithm)"
        else:
            txt = f"{fps_s} fps   ({mde:,} MDE/s · {ms:,.1f} ms)"
        ax.text(fps + xmax * 0.012, i, txt, va="center", fontsize=9.5,
                color=(COL["HW"] if hw else "black"), fontweight=("bold" if hw else "normal"))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.52)                       # headroom for the labels
    ax.set_xlabel("frames per second  (linear scale)", fontsize=11)
    ax.set_title(f"{title}\n{sub}\nsolid bars bit-exact to golden {golden}", fontsize=12.5, pad=12)
    handles = [Patch(color=COL[c], label=l) for c, l in
               [("GPU", "GPU"), ("CPU", "CPU"), ("DSP", "DSP"), ("REF", "scalar reference")]]
    handles.append(Patch(facecolor="white", edgecolor=COL["HW"], hatch="///",
                         label="fixed-function OFA — own algorithm, placed by wall-clock"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.075),
              ncol=3, frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=.25)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.155)
    fig.savefig(f"docs/{fname}", dpi=150)
    print("wrote docs/" + fname)
