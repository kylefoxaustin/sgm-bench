#!/usr/bin/env python3
"""vpi_stereo_bench.py — timed VPI stereo runs on the Jetson OFA engines.

The instrument behind the OFA rows: same pipeline as NVIDIA's stereo_disparity
sample (Y16_ER -> block-linear for OFA), but the estimate itself is run
warm+reps times with a sync per rep, and the MEDIAN is printed. Throughput
only — NOT bit-exact to this repo's goldens; VPI's SGM is NVIDIA's own
implementation with its own census, penalties and confidence.

    python3 vpi_stereo_bench.py ofa left.pgm right.pgm --maxdisp 128 --downscale 1
"""
import vpi, numpy as np, time, sys
from argparse import ArgumentParser

def read_pgm(p):
    with open(p, 'rb') as f:
        assert f.readline().strip() == b'P5', 'P5 pgm only'
        line = f.readline()
        while line.startswith(b'#'): line = f.readline()
        w, h = map(int, line.split()); int(f.readline())
        return np.frombuffer(f.read(w*h), dtype=np.uint8).reshape(h, w).copy()

ap = ArgumentParser()
ap.add_argument('backend', choices=['cuda', 'ofa', 'ofa-pva-vic'])
ap.add_argument('left'); ap.add_argument('right')
ap.add_argument('--maxdisp', type=int, default=128)
ap.add_argument('--downscale', type=int, default=1)
ap.add_argument('--reps', type=int, default=20)
ap.add_argument('--warm', type=int, default=3)
a = ap.parse_args()

backend = {'cuda': vpi.Backend.CUDA, 'ofa': vpi.Backend.OFA,
           'ofa-pva-vic': vpi.Backend.OFA | vpi.Backend.PVA | vpi.Backend.VIC}[a.backend]
L, R = read_pgm(a.left), read_pgm(a.right)

stream = vpi.Stream()
with vpi.Backend.CUDA, stream:
    left  = vpi.asimage(L).convert(vpi.Format.Y16_ER, scale=1)
    right = vpi.asimage(R).convert(vpi.Format.Y16_ER, scale=1)
if 'ofa' in a.backend:
    with vpi.Backend.VIC, stream:
        left  = left.convert(vpi.Format.Y16_ER_BL)
        right = right.convert(vpi.Format.Y16_ER_BL)
stream.sync()

times = []
for i in range(a.warm + a.reps):
    t0 = time.perf_counter()
    with stream, backend:
        disp = vpi.stereodisp(left, right, downscale=a.downscale, out_confmap=None,
                              window=5, maxdisp=a.maxdisp, includediagonals=True, numpasses=3)
    stream.sync()
    t1 = time.perf_counter()
    if i >= a.warm: times.append((t1 - t0) * 1000.0)

times.sort()
med = times[len(times)//2]
W, H = L.shape[1], L.shape[0]
ow, oh = (W + a.downscale - 1)//a.downscale, (H + a.downscale - 1)//a.downscale
print(f"vpi_stereo {a.backend}  in {W}x{H} out {ow}x{oh} maxdisp={a.maxdisp} "
      f"downscale={a.downscale}  median {med:.2f} ms  fps {1000.0/med:.2f}  "
      f"(n={a.reps}, spread {times[0]:.2f}-{times[-1]:.2f})")
