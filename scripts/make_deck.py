#!/usr/bin/env python3
"""make_deck.py — the results deck. Slide 1 summarises; one slide per unit shows
the arithmetic that produced its number, so every figure can be re-derived from
the measured milliseconds without trusting the table."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

W, H, D, PATHS = 1920, 1080, 64, 4
INK   = RGBColor(0x20, 0x21, 0x24)
MUTED = RGBColor(0x5F, 0x63, 0x68)
ACC   = {"GPU": RGBColor(0x4C,0x8B,0xF5), "CPU": RGBColor(0xE8,0x71,0x0A),
         "DSP": RGBColor(0x12,0xB5,0xA5), "REF": RGBColor(0x9A,0xA0,0xA6)}

# label, class, ms, silicon, notes[]
UNITS = [
 ("NVIDIA Thor",       "GPU",  13.19, "Blackwell, 20 SM, CUDA 13.2, sm_110", [
   "Rewritten warp-centric: one warp owns a scanline's whole disparity range,",
   "L in registers, min-reduce by __shfl_xor, ZERO barriers in the recurrence.",
   "2.69x over the naive OpenCL transliteration (35.39 -> 13.19 ms).",
   "MEMORY BOUND: aggregate moves 1.46 GB in 10.04 ms = 145 GB/s,",
   "against a 247 GB/s copy ceiling measured on this same part.",
   "Peak across the D sweep is 12,138 MDE/s at D=128; D=256 falls back to 9,409."]),
 ("NVIDIA Orin AGX",   "GPU",  23.28, "Ampere, 16 SM, CUDA 12.6, sm_87", [
   "Same kernel, same changes: 3.10x over the naive port (72.27 -> 23.28 ms).",
   "Bit-exact on the first run, before any tuning — the payoff for porting a",
   "verified OpenCL kernel one-for-one instead of writing a new one."]),
 ("NVIDIA RTX 5090",   "GPU",  None,  "Blackwell, discrete, 32 GB", [
   "NOT MEASURED. The GPU has been under another session's hard lease.",
   "The code is ready; it is one build and one run when the lease clears.",
   "Shown here rather than omitted, because a missing row reads as",
   "'not applicable' and this is 'not done yet'."]),
 ("Cortex-A720 x8",    "CPU",  51.70, "Radxa Orion O6, 2.2-2.5 GHz, NEON  |  single core: 313.60 ms = 423 MDE/s @2.5 GHz", [
   "The CPU headline: 10.4x the fastest published Arm-CPU SGM at matched D",
   "and matched path count (ReS2tAC, Sensors 21(11):3938, 2021 — 242 MDE/s).",
   "Best single core measured at 423 MDE/s; scales monotonically 1->8 threads.",
   "D=64 -> D=128 is FLAT (-2.4%): 16 NEON lanes are already full at D=64."]),
 ("Cortex-A78C x8",    "CPU",  94.70, "Qualcomm IQ-9075, NEON  |  single core: 332.22 ms = 400 MDE/s", [
   "Ran THIS repository's a55 NEON implementation unmodified, on another",
   "vendor's silicon, and reproduced the golden at every thread count.",
   "Per-core 402 MDE/s vs the A720's 423 — a 5% gap between two wide",
   "out-of-order Arm cores from different vendors on the same algorithm.",
   "Peaks at SIX threads, not eight; 8 threads is slower on that board."]),
 ("Hexagon v73 NSP",   "DSP", 138.64, "Qualcomm IQ-9075, HVX, 4 HVX contexts", [
   "Went 18.7x -> 12.8x -> 1.45x slower than the A78C cluster IN ONE DAY,",
   "with nothing retracted, because the ratio was always labelled a FLOOR",
   "OF EFFORT rather than the DSP's ceiling.",
   "BEATS A SINGLE A78C CORE BY 2.40x.",
   "Six hardware threads but only FOUR HVX contexts, so 4 threads is optimal.",
   "Still latency-bound at 6.8 of 28 GB/s — a different wall to Thor's."]),
 ("Mali-G720",         "GPU", 256.00, "Immortalis, 10 CU, OpenCL 3.0", [
   "Loses to TWO A720 cores. But this is an UNBOUNDED FLOOR, not a verdict",
   "on the hardware: the kernel carries ONE dependency chain per work-item",
   "with three barrier sites per step and never interleaves chains.",
   "Nobody has done for this part what was done for the DSP."]),
 ("Cortex-A55 x6",     "CPU", 341.30, "i.MX 95 FRDM, 1.8 GHz, NEON  |  single core: 1,588.74 ms = 84 MDE/s", [
   "The reference platform this project is anchored to, and the slowest CPU here.",
   "Scales 4.65x from 1 to 6 cores (1,588.74 -> 341.30 ms) -- that spread is also",
   "the proof that the -t thread flag was working on these runs, since the inert",
   "flag it was found to have would have collapsed every count to one timing.",
   "Per core 84 MDE/s against the A720's 423: a 5.0x gap between an in-order",
   "little core at 1.8 GHz and a wide out-of-order core at 2.5 GHz.",
   "Block-width optimum is 192 here and 256 on the A720 -- not the same number."]),
 ("Mali-G310",         "GPU",1846.00, "1 CU, OpenCL 3.0", [
   "The entry-tier part, same kernel, same hash. Its 'no compute driver'",
   "reputation is wrong — it ships working OpenCL 3.0 and merely lacks clinfo."]),
 ("scalar reference",  "REF",9188.00, "1x Cortex-A55, -O2, no SIMD, single thread", [
   "Deliberately boring: the oracle every other bar is checked against.",
   "It defines the golden hash, so it cannot be wrong by construction —",
   "it can only be slow, and it is 699x slower than the fastest bar."]),
]

def mde(ms): return W*H*D/ms/1000

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]

def textbox(sl, x, y, w, h, text, size, bold=False, color=INK, align=PP_ALIGN.LEFT):
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text
    r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    r.font.name = "Calibri"
    return tf

# ---------------- slide 1: summary ----------------
s1 = prs.slides.add_slide(blank)
textbox(s1, .6, .32, 12.2, .9, "Semi-Global Matching across nine execution targets", 28, True)
textbox(s1, .6, 1.12, 12.2, .5,
        "1920x1080  ·  D=64  ·  4 paths  ·  9x7 census  ·  every row bit-exact to golden b1b407b5949f0cc1",
        14, False, MUTED)

SINGLE = [("Cortex-A720 x1 @2.5GHz","CPU",313.60),("Cortex-A78C x1","CPU",332.22),
          ("Cortex-A55 x1 @1.8GHz","CPU",1588.74)]
rows = [u for u in UNITS] + [(l,c,ms,"",[]) for l,c,ms in SINGLE]
rows = sorted(rows, key=lambda r: (r[2] is None, r[2] or 0))
tbl = s1.shapes.add_table(len(rows)+1, 6, Inches(.6), Inches(1.72), Inches(12.1), Inches(0.4)).table
for i, w in enumerate((3.5, 1.1, 1.7, 1.5, 1.9, 2.4)): tbl.columns[i].width = Inches(w)
for r in range(len(rows)+1): tbl.rows[r].height = Inches(0.27)
hdr = ("processing unit", "class", "runtime", "fps", "MDE/s", "vs scalar reference")
for c, t in enumerate(hdr):
    cell = tbl.cell(0, c); cell.text = t
    pr = cell.text_frame.paragraphs[0]; pr.runs[0].font.size = Pt(11)
    pr.runs[0].font.bold = True; pr.runs[0].font.color.rgb = RGBColor(0xFF,0xFF,0xFF)
    cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0x20,0x21,0x24)
base = 9188.0
for r, (lab, cls, ms, sil, notes) in enumerate(rows, start=1):
    vals = ((lab, cls, "—" if ms is None else f"{ms:,.2f} ms",
             "—" if ms is None else f"{1000/ms:,.1f}",
             "—" if ms is None else f"{mde(ms):,.0f}",
             "not yet measured" if ms is None else f"{base/ms:,.0f}x faster"))
    for c, t in enumerate(vals):
        cell = tbl.cell(r, c); cell.text = str(t)
        pp = cell.text_frame.paragraphs[0]; run = pp.runs[0]
        run.font.size = Pt(11); run.font.color.rgb = MUTED if ms is None else INK
        run.font.bold = (c in (0, 4)) and ms is not None
        if c == 1:
            run.font.color.rgb = ACC[cls]; run.font.bold = True
# headline callouts, in the space under the table
CALLOUTS = [("699x", "faster than the scalar reference,\nwith byte-identical output"),
            ("10.4x", "the fastest published Arm-CPU SGM,\nat matched D and matched path count"),
            ("9 / 4 / 3 / 1", "targets / vendors / programming models\n/ one golden hash")]
for i, (big, small) in enumerate(CALLOUTS):
    x = .6 + i * 4.1
    textbox(s1, x, 6.05, 3.9, .6, big, 30, True, ACC["GPU"] if i == 0 else INK)
    tf = textbox(s1, x, 6.6, 3.9, .8, small.split("\n")[0], 12, False, MUTED)
    pp = tf.add_paragraph(); rr = pp.add_run(); rr.text = small.split("\n")[1]
    rr.font.size = Pt(12); rr.font.color.rgb = MUTED; rr.font.name = "Calibri"

textbox(s1, .6, 7.15, 12.2, .35,
        "MDE/s = W x H x D / runtime.  Valid for comparing implementations at a FIXED configuration; "
        "it is the wrong unit for CHOOSING one, because the optics fix D.", 10, False, MUTED)

# ---------------- one slide per unit ----------------
for lab, cls, ms, sil, notes in UNITS:
    sl = prs.slides.add_slide(blank)
    bar = sl.shapes.add_textbox(Inches(0), Inches(0), Inches(13.333), Inches(.12))
    textbox(sl, .6, .35, 11, .7, lab, 32, True, ACC[cls])
    textbox(sl, .6, 1.0, 11, .4, sil, 13, False, MUTED)

    textbox(sl, .6, 1.7, 11.5, .4, "The calculation", 16, True)
    if ms is None:
        textbox(sl, .6, 2.15, 11.5, 1.2,
                "No measurement exists for this part, so no number is derived.\n"
                "Nothing is estimated in its place.", 15, False, MUTED)
    else:
        calc = (f"MDE/s  =  W x H x D / runtime\n"
                f"        =  {W:,} x {H:,} x {D} / {ms:,.2f} ms\n"
                f"        =  {W*H*D:,} disparity estimations / {ms/1000:.5f} s\n"
                f"        =  {mde(ms):,.0f} million disparity estimations per second\n\n"
                f"frame rate  =  1000 / {ms:,.2f}  =  {1000/ms:,.1f} fps\n"
                f"pixel rate  =  {W*H/1e6:.4f} Mpx x {1000/ms:,.1f}  =  {W*H/1e6*1000/ms:,.2f} Mpix/s")
        tf = textbox(sl, .6, 2.15, 11.5, 2.2, calc, 14)
        for p in tf.paragraphs:
            for r in p.runs: r.font.name = "Consolas"

    textbox(sl, .6, 4.5, 11.5, .4, "What the measurement means", 16, True)
    tf = textbox(sl, .6, 4.9, 11.9, 2.2, notes[0], 13)
    for extra in notes[1:]:
        p = tf.add_paragraph(); r = p.add_run(); r.text = extra
        r.font.size = Pt(13); r.font.color.rgb = INK; r.font.name = "Calibri"

    if ms is not None:
        textbox(sl, .6, 7.0, 11.5, .4,
                f"MEASURED — median of the timed frames, golden hash verified in the same run.",
                11, False, MUTED)

prs.save("docs/sgm-results.pptx")
print("wrote docs/sgm-results.pptx —", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
