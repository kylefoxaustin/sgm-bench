#!/usr/bin/env python3
"""prep_real.py — turn a Middlebury 2014 scene into this benchmark's inputs.

REAL IMAGERY, not the synthetic generator. Everything in this repo until now was
measured on one procedurally generated scene; SGM's control flow is
data-independent so the TIMINGS should carry over, but accuracy on real texture,
occlusion and untextured regions is a different question that synthetic data
cannot answer.

Middlebury ships full-resolution pairs with dense float ground truth and an
`ndisp` in calib.txt. Downscaling by s scales disparity by s, so the scale is
chosen to bring ndisp under the SGM_D being tested. The ground truth is
rescaled the same way.

The scene files are NOT committed -- they are Middlebury's, several MB each, and
redistributing them is not ours to do. scripts/fetch_middlebury.sh pulls them.

Usage: prep_real.py SCENE_DIR OUT_DIR SCALE_DENOM
"""
import sys, struct, numpy as np
from PIL import Image

def read_pfm(path):
    with open(path, 'rb') as f:
        if f.readline().rstrip() != b'Pf': raise ValueError("not a greyscale PFM")
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        d = np.frombuffer(f.read(w*h*4), dtype='<f4' if scale < 0 else '>f4')
        return np.flipud(d.reshape(h, w))          # PFM is bottom-up

def main():
    scene, out, den = sys.argv[1], sys.argv[2], int(sys.argv[3])
    calib = dict(l.split('=',1) for l in open(f"{scene}/calib.txt").read().splitlines() if '=' in l)
    ndisp = int(calib['ndisp'])
    W, H = int(calib['width']), int(calib['height'])
    w, h = W//den, H//den
    print(f"{scene}: {W}x{H} ndisp={ndisp}  ->  {w}x{h} ndisp~{ndisp/den:.0f}")

    for src, dst in (("im0.png","left"), ("im1.png","right")):
        im = Image.open(f"{scene}/{src}").convert("L").resize((w,h), Image.LANCZOS)
        im.save(f"{out}/{dst}.pgm")

    gt = read_pfm(f"{scene}/disp0.pfm")
    gt = np.array(Image.fromarray(gt).resize((w,h), Image.NEAREST)) / den   # disparity scales with size
    valid = np.isfinite(gt)
    q = np.clip(np.nan_to_num(gt, nan=0, posinf=0), 0, 255).astype(np.uint8)
    Image.fromarray(q).save(f"{out}/gt.pgm")
    np.save(f"{out}/gt_valid.npy", valid)
    np.save(f"{out}/gt_float.npy", np.nan_to_num(gt, nan=-1, posinf=-1))
    print(f"  ground truth: valid {100*valid.mean():.1f}% of pixels, "
          f"max {np.nanmax(gt[valid]):.1f}, mean {np.nanmean(gt[valid]):.1f}")
    print(f"  -> needs SGM_D >= {int(np.ceil(np.nanmax(gt[valid])))+1}")

main()
