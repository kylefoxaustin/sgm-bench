#!/usr/bin/env python3
"""make_hw_chart.py — dedicated stereo hardware vs our software, matched config.

Separate figure from the main one on purpose. OFA only accepts maxDisparity of
128 or 256, so this comparison lives at D=128 and cannot be mixed into a D=64
chart without quietly comparing different amounts of work.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
W, H, D = 1920, 1080, 128

# (board, label, ms, kind)
ROWS = [
 ("NVIDIA Thor", "OFA hardware SGM engine",      9.345, "hw"),
 ("NVIDIA Thor", "our CUDA (bit-exact)",        21.860, "sw"),
 ("NVIDIA Orin AGX", "our CUDA (bit-exact)",    33.160, "sw"),
 ("NVIDIA Orin AGX", "OFA hardware SGM engine", 72.637, "hw"),
]
C = {"hw": "#7B4FD1", "sw": "#4C8BF5"}
fig, ax = plt.subplots(figsize=(11, 5.0))
labels = [f"{b}\n{l}" for b, l, _, _ in ROWS]
for i, (b, l, ms, k) in enumerate(ROWS):
    mde = W*H*D/ms/1000
    ax.barh(i, mde, color=C[k])
    ax.text(mde*1.02, i, f"  {mde:,.0f} MDE/s   ({ms:,.2f} ms)", va="center", fontsize=10)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis(); ax.set_xlim(0, 36000)
ax.set_xlabel("MDE/s at 1920×1080, D=128", fontsize=10)
ax.set_title("Dedicated stereo hardware vs. our software, same chip, same resolution and disparity range\n"
             "the hardware engine wins on Thor by 2.34× and LOSES on Orin by 2.19×\n"
             "THROUGHPUT ONLY — OFA is not bit-exact to our golden and its accuracy is unmeasured",
             fontsize=11, pad=12)
ax.legend(handles=[Patch(color=C["hw"], label="NVIDIA OFA — fixed-function SGM engine (not bit-exact to our golden)"),
                   Patch(color=C["sw"], label="our CUDA kernel — bit-exact to the golden hash")],
          loc="lower right", frameon=False, fontsize=9)
ax.grid(axis="x", alpha=.25)
for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("docs/hardware-vs-software.png", dpi=150)
print("wrote docs/hardware-vs-software.png")
