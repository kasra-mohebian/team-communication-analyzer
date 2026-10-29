"""
make_slides_v6.py — Iteration 6 Scientific Validation presentation.

White, academic theme. 8 slides covering:
  IAA study design & results · Model sanity check · What we now claim.

    python src/make_slides_v6.py   →   outputs/ITERATION6_VALIDATION.pptx
"""
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.chart.data import ChartData
import config

OUT    = str(config.OUTPUTS / "ITERATION6_VALIDATION.pptx")
IAA    = config.RESULTS / "annotation_agreement.json"
INSAMP = config.RESULTS / "insample_check.json"

with open(IAA)    as f: iaa    = json.load(f)
with open(INSAMP) as f: insamp = json.load(f)

# ── palette ───────────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x18, 0x23, 0x3A)
ACCENT = RGBColor(0x15, 0x52, 0xA0)
GREEN  = RGBColor(0x0D, 0x6B, 0x50)
AMBER  = RGBColor(0xC0, 0x70, 0x0A)
GREY   = RGBColor(0x5A, 0x63, 0x75)
LGREY  = RGBColor(0x90, 0x98, 0xA8)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT  = RGBColor(0xF0, 0xF2, 0xF5)
LTBLUE = RGBColor(0xDC, 0xE8, 0xFB)
LTGRN  = RGBColor(0xE0, 0xF5, 0xEE)
LTAMB  = RGBColor(0xFD, 0xF1, 0xE0)
BORDER = RGBColor(0xD2, 0xD7, 0xE0)
FONT   = "Calibri"

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK  = prs.slide_layouts[6]

N = 0
def number():
    global N; N += 1; return N

# ── primitives ────────────────────────────────────────────────────────────────
def slide():
    s = prs.slides.add_slide(BLANK)
    # white background
    bg = s.background.fill
    bg.solid(); bg.fore_color.rgb = WHITE
    return s

def rect(s, x, y, w, h, fill, line_color=None, line_width=Pt(0.75), radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    sp = s.shapes.add_shape(shape_type, x, y, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line_color:
        sp.line.color.rgb = line_color
        sp.line.width = line_width
    else:
        sp.line.fill.background()
    sp.shadow.inherit = False
    if radius:
        sp.adjustments[0] = 0.04
    return sp

def txt(s, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
        sa=3, ls=1.1):
    """paras: list of [(text, size, bold, color), ...]"""
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Pt(2)
    tf.margin_top  = tf.margin_bottom = Pt(2)
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(sa); p.line_spacing = ls
        for (t, sz, b, c) in para:
            r = p.add_run(); r.text = t
            r.font.size = Pt(sz); r.font.bold = b
            r.font.color.rgb = c; r.font.name = FONT
    return tb

def header(s, kicker, title):
    rect(s, 0, 0, SW, Inches(0.1), ACCENT)
    txt(s, Inches(0.6), Inches(0.18), Inches(12.1), Inches(0.35),
        [[(kicker.upper(), 9.5, True, ACCENT)]])
    txt(s, Inches(0.6), Inches(0.48), Inches(12.1), Inches(0.82),
        [[(title, 24, True, NAVY)]])
    rect(s, Inches(0.6), Inches(1.38), Inches(12.1), Pt(1), BORDER)

def footer(s, n):
    txt(s, Inches(0.6), Inches(7.1), Inches(10), Inches(0.3),
        [[("SocialCompass  ·  Iteration 6  ·  Scientific Validation", 8, False, LGREY)]])
    txt(s, Inches(12.1), Inches(7.1), Inches(0.8), Inches(0.3),
        [[(str(n), 8, False, LGREY)]], align=PP_ALIGN.RIGHT)

def info_box(s, x, y, w, h, text_paras, bg=LIGHT, border=BORDER):
    rect(s, x, y, w, h, bg, line_color=border)
    txt(s, x + Pt(8), y + Pt(6), w - Pt(16), h - Pt(12), text_paras)

def stat_tile(s, x, y, w, h, value, label, sub=None,
              val_color=ACCENT, bg=LIGHT):
    rect(s, x, y, w, h, bg, line_color=BORDER)
    txt(s, x + Pt(10), y + Pt(8),  w - Pt(20), Inches(0.65),
        [[(value, 34, True, val_color)]])
    txt(s, x + Pt(10), y + Inches(0.72), w - Pt(20), Inches(0.35),
        [[(label, 10, True, NAVY)]])
    if sub:
        txt(s, x + Pt(10), y + Inches(1.05), w - Pt(20), Inches(0.28),
            [[(sub, 9, False, GREY)]])

def tbl(s, x, y, w, rows, col_fracs,
        row_h=Inches(0.37), head_bg=ACCENT, fsize=10):
    nr, nc = len(rows), len(col_fracs)
    t = s.shapes.add_table(nr, nc, x, y, w, int(row_h * nr)).table
    for ci, frac in enumerate(col_fracs):
        t.columns[ci].width = int(w * frac)
    for ri, row in enumerate(rows):
        for ci, cell_val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.margin_left = cell.margin_right = Pt(5)
            cell.margin_top  = cell.margin_bottom = Pt(3)
            p = cell.text_frame.paragraphs[0]
            run = p.add_run(); run.text = str(cell_val)
            run.font.name = FONT
            if ri == 0:
                run.font.bold = True
                run.font.size = Pt(fsize - 0.5)
                run.font.color.rgb = WHITE
                cell.fill.solid(); cell.fill.fore_color.rgb = head_bg
            else:
                run.font.size = Pt(fsize)
                run.font.color.rgb = NAVY
                bg = WHITE if ri % 2 == 1 else LIGHT
                cell.fill.solid(); cell.fill.fore_color.rgb = bg
    return t

def column_chart(s, x, y, w, h, categories, series_list, max_val=1.0, fmt='0%'):
    """series_list: [(name, data_tuple, color), ...]"""
    cd = ChartData()
    cd.categories = categories
    for (name, data, _) in series_list:
        cd.add_series(name, data)
    cf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, w, h, cd)
    chart = cf.chart
    chart.has_title = False
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(9)
    chart.legend.font.color.rgb = GREY
    # series colors
    for i, (_, _, color) in enumerate(series_list):
        ser = chart.series[i]
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = color
        ser.format.line.fill.background()
    # value axis
    va = chart.value_axis
    va.maximum_scale = max_val
    va.minimum_scale = 0.0
    va.number_format = fmt
    va.number_format_is_linked = False
    va.tick_labels.font.size = Pt(8)
    va.tick_labels.font.color.rgb = GREY
    va.has_major_gridlines = True
    # category axis
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(9)
    ca.tick_labels.font.color.rgb = GREY
    ca.tick_labels.font.bold = False
    # data labels
    plot = chart.plots[0]
    plot.has_data_labels = True
    plot.data_labels.number_format = fmt
    plot.data_labels.font.size = Pt(8)
    plot.data_labels.font.color.rgb = NAVY
    return chart

def bar_chart(s, x, y, w, h, categories, series_list, max_val=1.0, fmt='0%'):
    """Horizontal bar chart."""
    cd = ChartData()
    cd.categories = categories
    for (name, data, _) in series_list:
        cd.add_series(name, data)
    cf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, x, y, w, h, cd)
    chart = cf.chart
    chart.has_title = False
    chart.has_legend = len(series_list) > 1
    if chart.has_legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(9)
    for i, (_, _, color) in enumerate(series_list):
        ser = chart.series[i]
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = color
        ser.format.line.fill.background()
    va = chart.value_axis
    va.maximum_scale = max_val
    va.minimum_scale = 0.0
    va.number_format = fmt
    va.number_format_is_linked = False
    va.tick_labels.font.size = Pt(8)
    va.tick_labels.font.color.rgb = GREY
    va.has_major_gridlines = True
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(9.5)
    ca.tick_labels.font.color.rgb = NAVY
    plot = chart.plots[0]
    plot.has_data_labels = True
    plot.data_labels.number_format = fmt
    plot.data_labels.font.size = Pt(8.5)
    plot.data_labels.font.color.rgb = NAVY
    return chart

# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Cover
# ═════════════════════════════════════════════════════════════════════════════
s = slide()
n = number()
rect(s, 0, 0, SW, Inches(0.1), ACCENT)

# kicker
txt(s, Inches(0.7), Inches(0.32), Inches(11.5), Inches(0.35),
    [[("SOCIALCOMPASS  ·  ITERATION 6", 10, True, GREY)]])

# accent bar
rect(s, Inches(0.7), Inches(0.82), Inches(0.5), Pt(3), ACCENT)

# main title
txt(s, Inches(0.7), Inches(1.0), Inches(9.5), Inches(1.4),
    [[("Scientific Validation\nof the Classifier", 38, True, NAVY)]])

# subtitle
txt(s, Inches(0.7), Inches(2.55), Inches(9.5), Inches(0.5),
    [[("Two independent checks — a human annotation study and a model sanity check —\n"
       "providing scientific backing for the archetype labels and the trained model.", 14, False, GREY)]])

# tag boxes
tags = [("Human IAA Study", LTBLUE, ACCENT),
        ("Model Sanity Check", LTGRN, GREEN),
        ("What We Now Claim", LTAMB, AMBER)]
tx = Inches(0.7)
for (tag, bg, fg) in tags:
    w_tag = Inches(2.55)
    rect(s, tx, Inches(3.4), w_tag, Inches(0.42), bg, line_color=fg)
    txt(s, tx + Pt(6), Inches(3.4) + Pt(5), w_tag - Pt(12), Inches(0.35),
        [[(tag, 11, True, fg)]])
    tx += w_tag + Inches(0.18)

# bottom stat line
txt(s, Inches(0.7), Inches(6.9), Inches(11), Inches(0.35),
    [[("5 archetypes  ·  300 labeled users  ·  2 validation passes  ·  50 IAA samples", 9, False, LGREY)]])
footer(s, n)

# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — System overview
# ═════════════════════════════════════════════════════════════════════════════
s = slide(); n = number()
header(s, "The System", "From raw chat to role prediction")

# pipeline boxes
steps = [
    ("Input",       "Team chats\n(Slack · WhatsApp)"),
    ("Features",    "13 graph metrics\n+ 17 text signals"),
    ("Labels",      "Opus soft labels\n300 users"),
    ("ML Training", "Random Forest\nFusion model"),
    ("Output",      "Archetype\nscores (soft)"),
]
box_w = Inches(2.05); box_h = Inches(1.05); gap = Inches(0.22)
start_x = Inches(0.55); y0 = Inches(1.6)
for i, (label, body) in enumerate(steps):
    bx = start_x + i * (box_w + gap)
    hi = (i == 2)  # highlight the labels box
    bg = LTBLUE if hi else LIGHT
    bc = ACCENT if hi else BORDER
    rect(s, bx, y0, box_w, box_h, bg, line_color=bc, line_width=Pt(1.2))
    txt(s, bx + Pt(6), y0 + Pt(5), box_w - Pt(12), Inches(0.32),
        [[(label.upper(), 8, True, ACCENT if hi else GREY)]])
    txt(s, bx + Pt(6), y0 + Inches(0.3), box_w - Pt(12), Inches(0.65),
        [[(body, 10, False, NAVY)]])
    if i < len(steps) - 1:
        ax = bx + box_w + Pt(4)
        txt(s, ax, y0 + Inches(0.35), gap - Pt(8), Inches(0.35),
            [[("→", 16, False, LGREY)]], align=PP_ALIGN.CENTER)

# two-column tables below
y_tbl = Inches(2.85)
col_l = Inches(0.6); col_r = Inches(7.0)
w_col = Inches(5.9)

# left: training data
txt(s, col_l, y_tbl, w_col, Inches(0.3),
    [[("TRAINING DATA", 8.5, True, GREY)]])
tbl(s, col_l, y_tbl + Inches(0.28), w_col,
    [["Dataset",     "Users", "Notes"],
     ["Slack",       "~1,639 total / labeled subset", "Public community chats"],
     ["Nankani CODERS", "labeled subset", "Student dev teams"],
     ["MBADA",       "labeled subset", "Open-source contributors"],
     ["Synthetic",   "53",   "Augmentation for rare classes, hand-verified"]],
    [0.22, 0.20, 0.58], fsize=10)

# right: model
txt(s, col_r, y_tbl, w_col, Inches(0.3),
    [[("MODEL ARCHITECTURE", 8.5, True, GREY)]])
tbl(s, col_r, y_tbl + Inches(0.28), w_col,
    [["View",          "Inputs",       "Detail"],
     ["Metric view",   "13 features",  "Graph + timing metrics, rank-normalised per dataset"],
     ["Text view",     "17 features",  "Empath, VADER, speech acts — scale-invariant"],
     ["Late fusion",   "Both views",   "0.5 metric + 0.5 text (CV-selected weight)"]],
    [0.20, 0.18, 0.62], fsize=10)

# callout box
info_box(s, Inches(0.6), Inches(6.05), Inches(12.1), Inches(0.72),
         [[("The open question:", 10, True, AMBER),
           ("  Opus labeled the training data — but is a single LLM annotator "
            "scientifically trustworthy? And did the model actually learn from those labels?",
            10, False, NAVY)]], bg=LTAMB, border=AMBER)
footer(s, n)

# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — IAA Study Design
# ═════════════════════════════════════════════════════════════════════════════
s = slide(); n = number()
header(s, "Validation I  —  Human Annotation Study", "Study design")

col_l = Inches(0.6); col_r = Inches(7.1)
w_l = Inches(6.1);  w_r = Inches(5.6)
y0  = Inches(1.55)

# left: numbered steps
steps_iaa = [
    ("1", "Population: 247 real users only",
     "Slack, Nankani, MBADA datasets. The 53 synthetic users were hand-labeled by us — "
     "including them would validate our own labels (circularity)."),
    ("2", "Stratified sample: 10 per archetype = 50 users",
     "Guarantees coverage of rare classes: butterfly (n=16) and capybara (n=19)."),
    ("3", "2 annotators, full overlap — both label all 50",
     "Gives us human↔human AND human↔Opus from the exact same sample."),
    ("4", "Blinded + input parity",
     "Annotators never see Opus labels. They receive the last 150 messages with [I]/[R] tags "
     "— exactly what Opus saw. Fair, apples-to-apples comparison."),
    ("5", "Task: split 100 points + confidence level",
     "Directly comparable to Opus soft labels and the project's Σmin metric."),
]
y = y0
for (num, title, body) in steps_iaa:
    bh = Inches(0.76)
    rect(s, col_l, y, w_l, bh, LIGHT, line_color=BORDER)
    txt(s, col_l + Pt(8), y + Pt(5), Inches(0.4), bh - Pt(10),
        [[(num, 18, False, LGREY)]], anchor=MSO_ANCHOR.MIDDLE)
    txt(s, col_l + Inches(0.5), y + Pt(5), w_l - Inches(0.6), Inches(0.3),
        [[(title, 10.5, True, NAVY)]])
    txt(s, col_l + Inches(0.5), y + Inches(0.3), w_l - Inches(0.6), Inches(0.44),
        [[(body, 9.5, False, GREY)]])
    y += bh + Pt(5)

# right: scientific basis + references
txt(s, col_r, y0, w_r, Inches(0.28),
    [[("SCIENTIFIC BASIS", 8.5, True, GREY)]])

rect(s, col_r, y0 + Inches(0.28), Pt(3), Inches(1.3), ACCENT)
txt(s, col_r + Inches(0.1), y0 + Inches(0.28), w_r - Inches(0.1), Inches(1.3),
    [[("If human↔model agreement ≥ human↔human agreement, the model operates "
       "within human inter-rater reliability and is justified as a weak supervisor "
       "for training.", 12, False, NAVY)]])

txt(s, col_r, y0 + Inches(1.72), w_r, Inches(0.28),
    [[("KEY DECISIONS", 8.5, True, GREY)]])
tbl(s, col_r, y0 + Inches(2.0), w_r,
    [["Decision",          "Choice",             "Reason"],
     ["Population",        "247 real users",     "Avoid circular validation of synthetic"],
     ["Sampling",          "Stratified 10/class","Cover rare classes"],
     ["Annotators",        "2, full overlap",    "Measure H↔H and H↔Opus from same set"],
     ["Blinding",          "Yes",                "Prevent anchoring on Opus labels"],
     ["Task format",       "100-point split",    "Directly comparable to Σmin metric"]],
    [0.28, 0.28, 0.44], fsize=9.5)

txt(s, col_r, y0 + Inches(4.2), w_r, Inches(0.55),
    [[("References: Gilardi et al. 2023 (PNAS) — LLM-as-annotator · "
       "Artstein & Poesio 2008 — IAA · Krippendorff 2004 — reliability theory",
       8.5, False, LGREY)]])
footer(s, n)

# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — IAA Results
# ═════════════════════════════════════════════════════════════════════════════
s = slide(); n = number()
header(s, "Validation I  —  Results",
       "Human↔Opus ≥ Human↔Human on every metric")

col_l = Inches(0.6); col_r = Inches(6.5)
w_l = Inches(5.5);  w_r = Inches(6.3)
y0  = Inches(1.58)

# left: metric table
txt(s, col_l, y0, w_l, Inches(0.28),
    [[("HEADLINE NUMBERS", 8.5, True, GREY)]])
ho  = iaa['headline']['human_vs_opus']
hh  = iaa['headline']['human_vs_human']
tbl(s, col_l, y0 + Inches(0.3), w_l,
    [["Metric",            "Human↔Opus",  "Human↔Human",  "Difference"],
     ["Soft-accuracy",     f"{ho['soft_accuracy']:.1%}",
                           f"{hh['soft_accuracy']:.1%}",  "+4.1 pp"],
     ["Cosine similarity", f"{ho['cosine']:.3f}",
                           f"{hh['cosine']:.3f}",          "+0.065"],
     ["Hard agreement",    f"{ho['hard_agreement']:.0%}",
                           f"{hh['hard_agreement']:.0%}",  "+4 pp"],
     ["Cohen's κ",         f"{ho['cohen_kappa']:.3f}",
                           f"{hh['cohen_kappa']:.3f}",     "+0.07"],
     ["Krippendorff α",
                           f"{iaa['krippendorff']['with_opus']['interval_avg']:.3f}",
                           f"{iaa['krippendorff']['humans_only']['interval_avg']:.3f}",
                           "+0.080"]],
    [0.34, 0.20, 0.22, 0.24], fsize=10)

# κ explanation box
info_box(s, col_l, y0 + Inches(2.65), w_l, Inches(1.0),
         [[("Why is κ in the 'fair' range?  ", 10, True, AMBER),
           ("Cohen's κ collapses soft distributions to a hard argmax, throwing away "
            "distributional information. Two annotators with nearly identical soft labels "
            "can disagree on the argmax for borderline users. "
            "Soft-accuracy and cosine are the appropriate primary metrics here.",
            9.5, False, NAVY)]], bg=LTAMB, border=AMBER)

# Krippendorff per-archetype
txt(s, col_l, y0 + Inches(3.82), w_l, Inches(0.28),
    [[("KRIPPENDORFF α PER ARCHETYPE", 8.5, True, GREY)]])
ka_w = iaa['krippendorff']['with_opus']['interval_per_dim']
ka_h = iaa['krippendorff']['humans_only']['interval_per_dim']
tbl(s, col_l, y0 + Inches(4.1), w_l,
    [["Archetype",    "With Opus",                    "Humans only"],
     ["🐝 Bee",       f"{ka_w['bee']:.3f}",          f"{ka_h['bee']:.3f}"],
     ["🐜 Ant",       f"{ka_w['ant']:.3f}",          f"{ka_h['ant']:.3f}"],
     ["🦋 Butterfly", f"{ka_w['butterfly']:.3f}",    f"{ka_h['butterfly']:.3f}"],
     ["🦫 Capybara",  f"{ka_w['capybara']:.3f}",     f"{ka_h['capybara']:.3f}"],
     ["🔴 Leech",     f"{ka_w['leech']:.3f}",        f"{ka_h['leech']:.3f}"]],
    [0.40, 0.30, 0.30], fsize=9.5, head_bg=GREY)

# right: grouped column chart
column_chart(
    s, col_r, y0 - Inches(0.05), w_r, Inches(5.2),
    categories=['Soft-Accuracy', 'Cosine Sim', 'Hard Agreement', "Cohen's κ"],
    series_list=[
        ("Human↔Opus",
         (ho['soft_accuracy'], ho['cosine'], ho['hard_agreement'], ho['cohen_kappa']),
         ACCENT),
        ("Human↔Human",
         (hh['soft_accuracy'], hh['cosine'], hh['hard_agreement'], hh['cohen_kappa']),
         AMBER),
    ],
    max_val=1.0, fmt='0%'
)
footer(s, n)

# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Sanity Check
# ═════════════════════════════════════════════════════════════════════════════
s = slide(); n = number()
header(s, "Validation II  —  Model Sanity Check",
       "In-sample vs held-out performance")

col_l = Inches(0.6); col_r = Inches(7.0)
w_l = Inches(6.0);  w_r = Inches(5.8)
y0  = Inches(1.58)

ins = insamp['insample']

# stat tiles
tile_w = Inches(2.7); tile_h = Inches(1.4)
stat_tile(s, col_l, y0, tile_w, tile_h,
          f"{ins['soft_accuracy']:.1%}", "In-sample soft-accuracy",
          f"{insamp['n_train']} training users", val_color=ACCENT)
stat_tile(s, col_l + tile_w + Inches(0.28), y0, tile_w, tile_h,
          f"{insamp['holdout_reference']['soft_accuracy']:.1%}",
          "Held-out soft-accuracy",
          "WhatsApp team — never seen during training", val_color=GREEN)

# gap note
gap_pct = ins['soft_accuracy'] - insamp['holdout_reference']['soft_accuracy']
info_box(s, col_l, y0 + Inches(1.5), w_l, Inches(0.55),
         [[("Gap:", 10, True, NAVY),
           (f"  {gap_pct:.1%}  — within expected range for Random Forest "
            "(bootstrap sampling means each tree sees ~63% of data).",
            10, False, GREY)]], bg=LIGHT, border=BORDER)

# additional numbers
txt(s, col_l, y0 + Inches(2.22), w_l, Inches(0.28),
    [[("ADDITIONAL IN-SAMPLE RESULTS", 8.5, True, GREY)]])
tbl(s, col_l, y0 + Inches(2.5), w_l,
    [["Metric",            "In-sample",            "Interpretation"],
     ["Cosine similarity", f"{ins['cosine']:.3f}",  "Near-perfect distribution match"],
     ["Hard accuracy",     f"{ins['hard_accuracy']:.0%}", "80% correct dominant archetype"],
     ["KL divergence",     f"{ins['kl_divergence']:.3f}", "Low — distributions close to labels"],
     ["Brier score",       f"{ins['brier_score']:.3f}",   "Low — well-calibrated probabilities"]],
    [0.36, 0.22, 0.42], fsize=10)

# underfit/overfit note
info_box(s, col_l, y0 + Inches(4.12), w_l, Inches(0.55),
         [[("If in-sample ≈ held-out →", 10, True, GREY),
           (" underfit.  ", 10, False, GREY),
           ("If held-out ≪ in-sample →", 10, True, GREY),
           (" overfit.  At 88% vs 74%, neither applies.", 10, False, NAVY)]],
         bg=LIGHT, border=BORDER)

# right: per-person held-out
txt(s, col_r, y0, w_r, Inches(0.28),
    [[("PER-PERSON HELD-OUT PREDICTIONS", 8.5, True, GREY)]])
ho_people = insamp['holdout_reference']['per_person']
rows_ho = [["Person", "Opus Label", "Model Prediction", "Match", "Soft-Acc"]]
for p in ho_people:
    match = "✓" if p['match'] else "✗"
    rows_ho.append([p['author'], p['opus'], p['model'],
                    match, f"{p['soft_accuracy']:.0%}"])
tbl(s, col_r, y0 + Inches(0.3), w_r, rows_ho,
    [0.22, 0.22, 0.28, 0.14, 0.14], fsize=10)

txt(s, col_r, y0 + Inches(2.6), w_r, Inches(0.28),
    [[("WHY NO FORMAL TRAIN/TEST SPLIT?", 8.5, True, GREY)]])
info_box(s, col_r, y0 + Inches(2.88), w_r, Inches(2.0),
         [[("5-fold × 10-repeat stratified CV  ", 10, True, NAVY),
           ("was used throughout model selection — statistically more "
            "stable than a single 80/20 split with n=300 and class imbalance.\n\n",
            10, False, GREY),
           ("External WhatsApp test  ", 10, True, NAVY),
           ("is a domain-transfer check — a different team, different platform — "
            "which is a stronger test of generalization than a random split from "
            "the same population.", 10, False, GREY)]], bg=LTBLUE, border=ACCENT)
footer(s, n)

# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Per-class depth
# ═════════════════════════════════════════════════════════════════════════════
s = slide(); n = number()
header(s, "Validation II  —  Depth", "In-sample performance per archetype")

col_l = Inches(0.6); col_r = Inches(7.1)
w_l = Inches(6.0);  w_r = Inches(5.7)
y0  = Inches(1.55)

# horizontal bar chart left
pcs = ins['per_class_soft_accuracy']
bar_chart(
    s, col_l, y0 - Inches(0.05), w_l, Inches(4.0),
    categories=['Bee', 'Ant', 'Butterfly', 'Capybara', 'Leech'],
    series_list=[("In-sample Soft-Accuracy",
                  (pcs['bee'], pcs['ant'], pcs['butterfly'],
                   pcs['capybara'], pcs['leech']),
                  ACCENT)],
    max_val=1.0, fmt='0%'
)

txt(s, col_l, y0 + Inches(4.08), w_l, Inches(0.5),
    [[("All five classes above 84% in-sample — no class collapsed. "
       "Butterfly and leech (minority classes, augmented with synthetic data) "
       "still learn successfully.", 9.5, False, GREY)]])

# right: F1 table
txt(s, col_r, y0, w_r, Inches(0.28),
    [[("PRECISION · RECALL · F1 (IN-SAMPLE, HARD LABEL)", 8.5, True, GREY)]])
pcf = ins['per_class_f1']
tbl(s, col_r, y0 + Inches(0.3), w_r,
    [["Archetype",    "n",  "Precision", "Recall", "F1"],
     ["🐝 Bee",       str(pcf['bee']['n']),
      f"{pcf['bee']['precision']:.3f}",
      f"{pcf['bee']['recall']:.3f}",
      f"{pcf['bee']['f1']:.3f}"],
     ["🐜 Ant",       str(pcf['ant']['n']),
      f"{pcf['ant']['precision']:.3f}",
      f"{pcf['ant']['recall']:.3f}",
      f"{pcf['ant']['f1']:.3f}"],
     ["🦋 Butterfly", str(pcf['butterfly']['n']),
      f"{pcf['butterfly']['precision']:.3f}",
      f"{pcf['butterfly']['recall']:.3f}",
      f"{pcf['butterfly']['f1']:.3f}"],
     ["🦫 Capybara",  str(pcf['capybara']['n']),
      f"{pcf['capybara']['precision']:.3f}",
      f"{pcf['capybara']['recall']:.3f}",
      f"{pcf['capybara']['f1']:.3f}"],
     ["🔴 Leech",     str(pcf['leech']['n']),
      f"{pcf['leech']['precision']:.3f}",
      f"{pcf['leech']['recall']:.3f}",
      f"{pcf['leech']['f1']:.3f}"]],
    [0.34, 0.10, 0.18, 0.18, 0.20], fsize=10.5)

info_box(s, col_r, y0 + Inches(2.88), w_r, Inches(0.82),
         [[("Bee has the lowest precision (0.706):", 10, True, AMBER),
           ("  the model over-predicts the majority class (104/300 = 35%). "
            "Consistent with the small-group scale bias documented in Iteration 4 "
            "and partially corrected by the text-view fusion.", 9.5, False, GREY)]],
         bg=LTAMB, border=AMBER)

info_box(s, col_r, y0 + Inches(3.82), w_r, Inches(0.75),
         [[("Leech precision 0.977:", 10, True, GREEN),
           ("  when the model says leech, it is almost always right. "
            "Recall 0.75 means some leeches are missed — "
            "likely near-tie profiles with bee.", 9.5, False, GREY)]],
         bg=LTGRN, border=GREEN)
footer(s, n)

# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — The Full Picture
# ═════════════════════════════════════════════════════════════════════════════
s = slide(); n = number()
header(s, "Putting It Together", "Three independent sources of evidence")

y0  = Inches(1.62)
col_l = Inches(0.6); col_r = Inches(7.25)
w_l = Inches(6.1);  w_r = Inches(5.5)

evidences = [
    ("Evidence 1 — Label validity", ACCENT, LTBLUE,
     "Human annotators agree with Opus as much as they agree with each other",
     f"Soft-acc: H↔Opus {ho['soft_accuracy']:.1%}  vs  H↔H {hh['soft_accuracy']:.1%}  →  Opus operates within human inter-rater reliability."),
    ("Evidence 2 — Model sanity", GREEN, LTGRN,
     "The model learned from the labels and generalizes to an unseen team",
     f"In-sample {ins['soft_accuracy']:.1%} → held-out {insamp['holdout_reference']['soft_accuracy']:.1%}  (gap: {gap_pct:.1%}).  3/5 hard-label matches on a different team, different platform."),
    ("Evidence 3 — Cross-validation", AMBER, LTAMB,
     "Model selection used 5-fold × 10-repeat CV — no look-ahead bias",
     "Fusion weight and model type both chosen by CV. External WhatsApp test is a domain-transfer check, not a held-out from the same population."),
]
y = y0
for (kicker_t, fg, bg, title_t, body_t) in evidences:
    bh = Inches(1.5)
    rect(s, col_l, y, w_l, bh, bg, line_color=fg, line_width=Pt(1.2))
    rect(s, col_l, y, Pt(4), bh, fg)
    txt(s, col_l + Inches(0.12), y + Pt(6), w_l - Inches(0.2), Inches(0.28),
        [[(kicker_t.upper(), 8.5, True, fg)]])
    txt(s, col_l + Inches(0.12), y + Inches(0.32), w_l - Inches(0.2), Inches(0.38),
        [[(title_t, 12, True, NAVY)]])
    txt(s, col_l + Inches(0.12), y + Inches(0.72), w_l - Inches(0.2), Inches(0.65),
        [[(body_t, 9.5, False, GREY)]])
    y += bh + Pt(6)

# right: what each covers
txt(s, col_r, y0, w_r, Inches(0.28),
    [[("WHAT EACH CHECK COVERS", 8.5, True, GREY)]])
tbl(s, col_r, y0 + Inches(0.3), w_r,
    [["Check",            "Question",              "Output"],
     ["IAA Study",        "Are the labels valid?",
      "H↔Opus ≥ H↔H on all metrics"],
     ["Sanity check",     "Did the model learn?",
      "In-sample 88.4% — yes, it learned"],
     ["Held-out test",    "Does it generalize?",
      "74% on an unseen team"],
     ["CV (5×10)",        "Is model selection sound?",
      "Fusion weight and architecture both CV-selected"],
     ["Domain shift",     "Is the team representative?",
      "Cohen's d checked per feature in Iteration 4"]],
    [0.24, 0.36, 0.40], fsize=10)

info_box(s, col_r, y0 + Inches(3.15), w_r, Inches(1.4),
         [[("Together these three checks constitute a defensible "
            "validation argument for an academic presentation:\n", 11, True, NAVY),
           ("(1) the training labels are trustworthy  ·  "
            "(2) the model actually learned from them  ·  "
            "(3) it generalizes beyond the training population", 10, False, GREY)]],
         bg=LTBLUE, border=ACCENT)
footer(s, n)

# ═════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — What We Claim
# ═════════════════════════════════════════════════════════════════════════════
s = slide(); n = number()
header(s, "Conclusion  —  Iteration 6", "What we can and cannot claim")

col_l = Inches(0.6); col_r = Inches(7.15)
w_l = Inches(5.9);  w_r = Inches(5.6)
y0  = Inches(1.58)

# confirmed
txt(s, col_l, y0, w_l, Inches(0.3),
    [[("CONFIRMED", 9, True, GREEN)]])
confirmed = [
    ("Opus labels are within human reliability.",
     f"H↔Opus soft-acc ({ho['soft_accuracy']:.1%}) ≥ H↔H ({hh['soft_accuracy']:.1%}) on all five metrics. "
     "The labeling step is scientifically justified as a weak supervisor."),
    ("The model learned from the labels.",
     f"In-sample soft-acc {ins['soft_accuracy']:.1%} confirms the model fits training data. "
     f"The gap to held-out ({insamp['holdout_reference']['soft_accuracy']:.1%}) "
     f"is {gap_pct:.1%} — consistent with Random Forest bootstrap behavior."),
    ("The model generalizes across contexts.",
     f"Tested on a different team and platform: soft-acc {insamp['holdout_reference']['soft_accuracy']:.1%}, "
     f"cosine {insamp['holdout_reference']['mean_cosine']:.3f}, "
     f"{int(insamp['holdout_reference']['top1_agreement']*5)}/5 hard-label matches."),
]
y = y0 + Inches(0.32)
for (title_c, body_c) in confirmed:
    bh = Inches(1.35)
    rect(s, col_l, y, w_l, bh, LTGRN, line_color=GREEN, line_width=Pt(1))
    txt(s, col_l + Pt(10), y + Pt(6), Inches(0.25), bh,
        [[("✓", 16, True, GREEN)]], anchor=MSO_ANCHOR.MIDDLE)
    txt(s, col_l + Pt(34), y + Pt(6), w_l - Pt(44), Inches(0.32),
        [[(title_c, 11, True, NAVY)]])
    txt(s, col_l + Pt(34), y + Inches(0.36), w_l - Pt(44), Inches(0.85),
        [[(body_c, 9.5, False, GREY)]])
    y += bh + Pt(5)

# not claimed
txt(s, col_r, y0, w_r, Inches(0.3),
    [[("NOT CLAIMED", 9, True, GREY)]])
not_claimed = [
    ("The IAA study does not validate the ML model.",
     "It validates the labels. Model performance is assessed separately "
     "by the held-out test. These are two independent claims."),
    ("Cohen's κ does not prove strong reliability.",
     "Fair-range κ reflects 5-way fuzzy annotation and the inherent ambiguity "
     "of behavioral archetypes, not model failure. Soft-accuracy and cosine "
     "are the appropriate primary metrics."),
    ("No formal train/test split existed.",
     "We used 5-fold × 10-repeat CV for model selection plus an external "
     "domain-transfer test. Acknowledged as a limitation; partially compensated "
     "by the CV design and the external team."),
]
y = y0 + Inches(0.32)
for (title_nc, body_nc) in not_claimed:
    bh = Inches(1.35)
    rect(s, col_r, y, w_r, bh, LIGHT, line_color=BORDER, line_width=Pt(0.75))
    txt(s, col_r + Pt(10), y + Pt(6), Inches(0.28), bh,
        [[("–", 16, False, LGREY)]], anchor=MSO_ANCHOR.MIDDLE)
    txt(s, col_r + Pt(36), y + Pt(6), w_r - Pt(46), Inches(0.32),
        [[(title_nc, 11, True, NAVY)]])
    txt(s, col_r + Pt(36), y + Inches(0.36), w_r - Pt(46), Inches(0.85),
        [[(body_nc, 9.5, False, GREY)]])
    y += bh + Pt(5)

txt(s, Inches(0.6), Inches(6.92), Inches(12.1), Inches(0.3),
    [[("Full results:  outputs/results/annotation_agreement.json  ·  "
       "insample_check.json  ·  iaa_report.html", 8, False, LGREY)]])
footer(s, n)

# ═════════════════════════════════════════════════════════════════════════════
prs.save(OUT)
print(f"✅  Saved → {OUT}  ({N} slides)")
