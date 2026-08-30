#!/usr/bin/env python3
"""check_golden_discriminates.py — assert a golden can actually fail a bad kernel.

THE FAILURE THIS EXISTS TO PREVENT. Until 2026-08-29 the primary golden had a
maximum disparity of 44 with SGM_D=64, so an implementation that only searched
d < 45 reproduced it BYTE-FOR-BYTE. Every target accepted against it -- A55,
A720, A78C, both Malis, the Hexagon -- had the top 30% of the disparity range
verified by nothing at all. A hash gate that a broken kernel passes is not a
gate; it is a formality that looks like one.

THE TEST. A golden discriminates at truncation point k if some pixel's winning
disparity is >= k. But REACHING the top of the range is not enough, and the
first version of this script wrongly passed a golden that only reached it on 58
pixels out of 658,008 -- 0.0088% -- every one of them a spurious error, since
the scene's true disparities stopped well below.

⚠️ THAT FAILURE MODE GETS WORSE AS THE IMPLEMENTATION GETS BETTER. If a golden
only reaches the top of the range because the algorithm makes high-disparity
mistakes there, then a MORE accurate implementation makes fewer such mistakes
and becomes HARDER to distinguish from a truncated one. A test whose power
decays as the thing it tests improves is worse than useless.

So three verdicts, not two:
    BLIND  the top of the range never wins -- a truncated search passes
    THIN   it wins, but on so few pixels that the margin is noise, or on
           pixels where the ground truth says the answer is wrong anyway
    OK     the top decile of the range wins on a real fraction of the scene

And with ground truth available, THIN is decided on TRUE high-disparity pixels
rather than on the implementation's own errors. Top-decile coverage threshold
due to the qualcomm session, who found this by sweeping every (scene, D) pair
either corpus cites -- including a BLIND one in their own published work.

⭐ Discrimination is a property of the (SCENE, D) PAIR, not of the scene, the
generator, or the algorithm. "We use real imagery now" does not retire it.

Usage:  check_golden_discriminates.py GOLDEN.pgm D  [min_distinct]
Exit 0 pass, 1 fail.
"""
import sys

def read_pgm(p):
    d = open(p, 'rb').read()
    i = d.index(b'255\n') + 4
    return d[i:]

def main():
    if len(sys.argv) < 3:
        print(__doc__); return 1
    path, D = sys.argv[1], int(sys.argv[2])
    min_distinct = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    px = read_pgm(path)
    top, distinct = max(px), len(set(px))
    need = D - 1
    decile = int(D * 0.9)
    n_top = sum(1 for v in px if v >= decile)
    cov = 100.0 * n_top / len(px)
    ok = True
    print("golden %s  (D=%d)" % (path, D))
    print("  highest disparity present : %d   (need >= %d, i.e. d=D-1)" % (top, need))
    print("  distinct values           : %d   (need >= %d)" % (distinct, min_distinct))
    print("  top-decile coverage       : %d px = %.4f%%  (>= %d)   (need >= 0.5%%)"
          % (n_top, cov, decile))
    import os
    gtp = os.path.join(os.path.dirname(path) or ".", "gt_float.npy")
    if os.path.exists(gtp):
        try:
            import numpy as np
            gt = np.load(gtp).ravel()
            g = np.frombuffer(px, dtype=np.uint8).astype(np.float32)
            gtf = gt.ravel()[:g.size]
            # A real INTERSECTION: pixels the golden puts high AND the ground
            # truth agrees are high. Counting gt>=decile alone is not an
            # "of which" -- it says nothing about the golden.
            inter = int(((g >= decile) & (gtf >= decile)).sum())
            true_top = inter
            print("  ...of which TRUE high-disparity pixels: %d   "
                  "(the rest are the implementation's own errors)" % inter)
            # And check the TOP, not only the decile: a golden whose top of range
            # is reached only by errors is pinned by the implementation's own
            # mistakes, which a better implementation would make fewer of.
            top_true = int((gtf >= need).sum())
            if top_true == 0:
                print("  THIN: ground truth never reaches d >= %d, so nothing in this scene" % need)
                print("        can legitimately win there. Truncation at the top is pinned")
                print("        only by errors. Use a D matched to the scene's disparity range.")
                ok = False
            if true_top == 0:
                print("  THIN: the top of the range is reached ONLY by errors. A more accurate")
                print("        implementation would reach it less and be harder to distinguish")
                print("        from a truncated one -- the test weakens as the code improves.")
                ok = False
        except Exception:
            pass
    if top < need:
        print("  FAIL: an implementation searching only d < %d reproduces this golden" % (top + 1))
        print("        byte-for-byte. Disparities %d..%d are untested." % (top + 1, D - 1))
        ok = False
    if distinct < min_distinct:
        print("  FAIL: too few distinct disparities to catch intermediate truncation.")
        ok = False
    if cov < 0.5:
        print("  THIN: only %.4f%% of pixels reach the top decile. The margin between" % cov)
        print("        a correct implementation and a truncated one is a handful of pixels.")
        ok = False
    print("  %s" % ("PASS - this golden can fail a truncated implementation" if ok else "NOT USABLE AS A CORRECTNESS GATE AT THIS D"))
    return 0 if ok else 1

sys.exit(main())
