#!/usr/bin/env python3
"""check_golden_discriminates.py — assert a golden can actually fail a bad kernel.

THE FAILURE THIS EXISTS TO PREVENT. Until 2026-08-29 the primary golden had a
maximum disparity of 44 with SGM_D=64, so an implementation that only searched
d < 45 reproduced it BYTE-FOR-BYTE. Every target accepted against it -- A55,
A720, A78C, both Malis, the Hexagon -- had the top 30% of the disparity range
verified by nothing at all. A hash gate that a broken kernel passes is not a
gate; it is a formality that looks like one.

THE TEST. A golden discriminates at truncation point k if some pixel's winning
disparity is >= k. Scan the golden's own output: the highest disparity it
contains is the highest truncation an implementation could get away with. We
require the golden to reach at least SGM_D-2, i.e. the top of the searchable
range, and to contain enough distinct values that intermediate truncations are
caught too.

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
    need = D - 2
    ok = True
    print("golden %s  (D=%d)" % (path, D))
    print("  highest disparity present : %d   (need >= %d)" % (top, need))
    print("  distinct values           : %d   (need >= %d)" % (distinct, min_distinct))
    if top < need:
        print("  FAIL: an implementation searching only d < %d reproduces this golden" % (top + 1))
        print("        byte-for-byte. Disparities %d..%d are untested." % (top + 1, D - 1))
        ok = False
    if distinct < min_distinct:
        print("  FAIL: too few distinct disparities to catch intermediate truncation.")
        ok = False
    print("  %s" % ("PASS - this golden can fail a truncated implementation" if ok else "FAIL"))
    return 0 if ok else 1

sys.exit(main())
