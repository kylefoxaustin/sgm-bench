#!/usr/bin/env python3
"""make_panel.py — a per-target visual: what the disparity map actually looks like.

A hash proves two targets agree. It does not show whether the ANSWER is any
good, and a table of MDE/s shows neither. This renders, for one target:

    left image | its disparity map | error against ground truth

Bad pixels (|d - gt| > 1) are red, correct ones grey, and pixels with no ground
truth are left dark. Because every target here produces byte-identical output,
the middle and right panels are the same for all of them -- which is the point:
the picture is the same, only the time differs.

Usage: make_panel.py SCENE_DIR DISPARITY.pgm "Target name" "13.16 ms · 10,085 MDE/s" OUT.png
"""
import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont

def pgm(p):
    d = open(p,'rb').read(); i = d.index(b'255\n')+4
    hdr = d[:i].split()
    w, h = int(hdr[1]), int(hdr[2])
    return np.frombuffer(d[i:i+w*h], dtype=np.uint8).reshape(h, w)

def main():
    scene, dpath, name, stat, out = sys.argv[1:6]
    left = pgm(f"{scene}/left.pgm"); disp = pgm(dpath)
    gt = np.load(f"{scene}/gt_float.npy"); valid = np.load(f"{scene}/gt_valid.npy")
    h, w = disp.shape

    err = np.abs(disp.astype(np.float32) - gt)
    # The leftmost D columns cannot be matched at all: x-d < 0 has no
    # correspondent in the right image. Middlebury evaluation excludes such
    # regions and so do we, rather than bank an error the algorithm is
    # structurally incapable of avoiding.
    D = int(sys.argv[6]) if len(sys.argv) > 6 else 128
    scoreable = valid.copy(); scoreable[:, :D] = False
    bad1 = (err > 1.0) & scoreable
    bad2 = (err > 2.0) & scoreable
    pct_bad = 100.0 * bad1.sum() / scoreable.sum()
    pct_bad2 = 100.0 * bad2.sum() / scoreable.sum()
    mae = err[scoreable].mean()
    bad = bad1

    # panel 1: the scene.  panel 2: disparity, turbo-ish.  panel 3: error map.
    p1 = np.stack([left]*3, -1)
    import matplotlib; matplotlib.use("Agg")
    from matplotlib import colormaps
    dn = disp.astype(np.float32) / max(disp.max(), 1)
    p2 = (colormaps["turbo"](dn)[..., :3] * 255).astype(np.uint8)
    p3 = np.stack([left//3]*3, -1)
    p3[bad] = [220, 40, 40]
    p3[~scoreable] = [20, 20, 20]      # no ground truth, or unmatchable border

    gap, top = 8, 46
    canvas = Image.new("RGB", (w*3 + gap*2, h + top + 30), (250, 250, 250))
    for i, p in enumerate((p1, p2, p3)):
        canvas.paste(Image.fromarray(p), (i*(w+gap), top))
    dr = ImageDraw.Draw(canvas)
    try:
        f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        f1 = f2 = ImageFont.load_default()
    dr.text((4, 6),  name, fill=(20,20,20), font=f1)
    dr.text((4, 30), stat, fill=(90,90,90), font=f2)
    for i, t in enumerate(("left image (Middlebury)", "disparity",
                           f"error > 1px in red — bad1 {pct_bad:.1f}%  bad2 {pct_bad2:.1f}%  MAE {mae:.2f}   (dark = excluded)")):
        dr.text((i*(w+gap)+4, h+top+6), t, fill=(90,90,90), font=f2)
    # Every target produces byte-identical output, so ONE picture serves them
    # all and the interesting variable is time. Render that as a strip rather
    # than shipping five copies of the same image with different captions.
    import os
    strip = os.environ.get("PANEL_TIMES")
    if strip:
        rows = [r.split("|") for r in strip.split(";") if r]
        bh = 30; extra = bh*len(rows) + 40
        c2 = Image.new("RGB", (canvas.width, canvas.height + extra), (250,250,250))
        c2.paste(canvas, (0,0)); dr2 = ImageDraw.Draw(c2)
        y0 = canvas.height + 8
        dr2.text((4, y0), "every target below produced this identical disparity map "
                          f"(hash {os.environ.get('PANEL_HASH','')}) — only the time differs",
                 fill=(20,20,20), font=f2)
        worst = max(float(r[1]) for r in rows)
        for i,(nm,ms,extra_s) in enumerate(rows):
            y = y0 + 26 + i*bh
            wpx = int((float(ms)/worst) * (canvas.width - 560))
            dr2.rectangle([300, y+5, 300+max(wpx,3), y+21], fill=(76,139,245))
            dr2.text((4, y+4), nm, fill=(20,20,20), font=f2)
            dr2.text((310+max(wpx,3), y+4), f"{float(ms):,.2f} ms   {extra_s}", fill=(60,60,60), font=f2)
        canvas = c2
    canvas.save(out)
    print(f"{out}  bad>1px {pct_bad:.1f}%  bad>2px {pct_bad2:.1f}%  MAE {mae:.2f}")

main()
