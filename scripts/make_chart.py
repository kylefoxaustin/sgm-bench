#!/usr/bin/env python3
"""make_chart.py — the results figure for the README.

Every bar is MEASURED at the shared configuration: 1920x1080, D=64, 4 paths,
9x7 census, bit-exact against the same golden hash. One unmeasured part is drawn
explicitly as unmeasured rather than left out, because a missing bar reads as
"not applicable" while a hatched one reads as "not done yet".
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# (label, class, ms, MDE/s) — MEASURED, 1080p D=64. None = not yet measured.
# Clock is shown only where it was actually read off the machine. The Jetson GPUs
# report [N/A] to nvidia-smi and have no readable devfreq node, and the A78C
# board was under another session's lease, so those are left blank rather than
# guessed -- an empty field beats a plausible number.
DATA = [
    ("NVIDIA Thor",             "GPU",  13.19, 10062),
    ("NVIDIA Orin AGX",         "GPU",  23.28,  5700),
    ("Cortex-A720 x8 @2.2-2.5GHz","CPU", 51.70,  2569),
    ("Cortex-A78C x8",          "CPU",  94.70,  1402),
    ("Hexagon v73 NSP @1.46GHz","DSP", 138.64,   957),
    ("Mali-G720 (10 CU)",       "GPU", 256.00,   518),
    ("Cortex-A720 x1 @2.5GHz",  "CPU", 313.60,   423),
    ("Cortex-A78C x1",          "CPU", 332.22,   400),
    ("Cortex-A55 x6 @1.8GHz",   "CPU", 341.30,   389),
    ("Cortex-A55 x1 @1.8GHz",   "CPU",1588.74,    84),
    ("Mali-G310 (1 CU)",        "GPU",1846.00,    72),
    ("scalar reference (1 core)","REF",9188.00,  14.4),
    ("NVIDIA RTX 5090",         "GPU",  None,   None),
]
COL = {"GPU": "#4C8BF5", "CPU": "#E8710A", "DSP": "#12B5A5", "REF": "#9AA0A6"}

rows = sorted(DATA, key=lambda r: (r[3] is None, -(r[3] or 0)))
labels = [r[0] for r in rows]
vals   = [r[3] if r[3] else 1 for r in rows]
y = range(len(rows))

fig, ax = plt.subplots(figsize=(11.5, 7.4))
for i, (lab, cls, ms, mde) in enumerate(rows):
    if mde is None:
        ax.barh(i, 12000, color="none", edgecolor=COL[cls], hatch="///", linewidth=1.4, alpha=.75)
        ax.text(35, i, "  NOT MEASURED — GPU held under another session's lease",
                va="center", ha="left", fontsize=9, color="#5f6368", style="italic")
    else:
        ax.barh(i, mde, color=COL[cls])
        ax.text(mde * 1.08, i, f"{mde:,.0f}   ({ms:,.1f} ms)", va="center", fontsize=9)

ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(10, 60000)
ax.set_xlabel("MDE/s  —  million disparity estimations per second  (log scale)", fontsize=10)
ax.set_title("Semi-Global Matching, 1920×1080, D=64, 4 paths, 9×7 census\n"
             "every bar bit-exact to the same golden hash b1b407b5949f0cc1",
             fontsize=12, pad=14)
ax.legend(handles=[Patch(color=COL[c], label=l) for c, l in
                   [("GPU","GPU"),("CPU","CPU"),("DSP","DSP"),("REF","scalar reference")]],
          loc="lower right", frameon=False, fontsize=9)
ax.grid(axis="x", alpha=.25, which="both")
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("docs/results.png", dpi=150)
print("wrote docs/results.png")
