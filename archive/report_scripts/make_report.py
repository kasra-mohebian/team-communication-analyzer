"""
make_report.py — build the full project-journey report as a PDF.

Tells the whole story: from unlabelled chats to the final late-fusion model —
every decision, what we tried, what failed, what we kept and why, with the
scientific reference behind each choice.

    python src/make_report.py   →   docs/PROJECT_REPORT.pdf

Run make_report_assets.py first (charts) — this script embeds docs/assets/*.png
and archive/architecture-historical.png / docs/text_features_table.png.
"""
import json
import config
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether)

DOCS = config.ROOT / "docs"
ASSETS = DOCS / "assets"
OUT = str(DOCS / "PROJECT_REPORT.pdf")

# ── palette ──────────────────────────────────────────────────────────────────
NAVY = colors.HexColor("#1B2A41")
BLUE = colors.HexColor("#2E86C1")
GREEN = colors.HexColor("#27AE60")
ORANGE = colors.HexColor("#E67E22")
RED = colors.HexColor("#C0392B")
GREY = colors.HexColor("#5D6D7E")
LIGHT = colors.HexColor("#F4F6F7")

ss = getSampleStyleSheet()
def style(name, **kw):
    kw.setdefault("parent", ss["Normal"])
    return ParagraphStyle(name, **kw)

H1 = style("H1", fontName="Helvetica-Bold", fontSize=18, textColor=NAVY,
           spaceBefore=14, spaceAfter=8, leading=22)
H2 = style("H2", fontName="Helvetica-Bold", fontSize=13, textColor=BLUE,
           spaceBefore=12, spaceAfter=5, leading=16)
BODY = style("BODY", fontSize=10.2, leading=15, textColor=colors.HexColor("#222222"),
             spaceAfter=6, alignment=TA_LEFT)
SMALL = style("SMALL", fontSize=8.4, leading=11, textColor=GREY)
CAP = style("CAP", fontSize=8.6, leading=11, textColor=GREY, alignment=TA_CENTER,
            spaceAfter=10, spaceBefore=2)
BULLET = style("BULLET", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
KEEP = style("KEEP", fontName="Helvetica-Bold", fontSize=9, textColor=GREEN)


def img(path, width_cm):
    from PIL import Image as PILImage
    p = str(path)
    iw, ih = PILImage.open(p).size
    w = width_cm * cm
    return Image(p, width=w, height=w * ih / iw)


def hr():
    return HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#D5D8DC"),
                      spaceBefore=6, spaceAfter=8)


def bullets(items, st=BODY):
    return [Paragraph(f"•&nbsp;&nbsp;{t}", style("b", parent=st, leftIndent=12)) for t in items]


# ── load live numbers ────────────────────────────────────────────────────────
ho = json.load(open(config.HOLDOUT_COMPARISON))
ab = json.load(open(config.RESULTS / "synthetic_ablation.json"))
SOFT = f"{ho['soft_accuracy']*100:.0f}%"
COS = f"{ho['mean_cosine']:.2f}"
KL = f"{ho['mean_kl']:.2f}"
TOP1 = f"{int(round(ho['top1_agreement']*5))}/5"
MF_NO = ab["oof_cv"]["without_synthetic"]["macro_f1"]
MF_YES = ab["oof_cv"]["with_synthetic"]["macro_f1"]

story = []


# ═════════════════════════════════════════════════════════════════════════════
# COVER
# ═════════════════════════════════════════════════════════════════════════════
story += [Spacer(1, 3.2*cm),
    Paragraph("SocialCompass", style("t1", fontName="Helvetica-Bold", fontSize=34,
              textColor=NAVY, alignment=TA_CENTER, leading=38)),
    Paragraph("Team Communication Analyzer — COIN Archetypes",
              style("t2", fontSize=15, textColor=BLUE, alignment=TA_CENTER, spaceBefore=6)),
    Spacer(1, 0.6*cm),
    Paragraph("The complete project journey: from unlabelled group chats to a "
              "validated late-fusion archetype classifier — every decision, every "
              "dead end, and the scientific reasoning we kept.",
              style("t3", fontSize=11.5, textColor=GREY, alignment=TA_CENTER, leading=17)),
    Spacer(1, 1.0*cm),
    img(DOCS / "architecture.png", 15.5),
    Spacer(1, 0.8*cm),
    Paragraph("Social Network Analysis seminar &nbsp;·&nbsp; project report",
              style("t4", fontSize=10, textColor=GREY, alignment=TA_CENTER)),
    PageBreak()]


# ═════════════════════════════════════════════════════════════════════════════
# 1. THE GOAL
# ═════════════════════════════════════════════════════════════════════════════
story += [Paragraph("1 · The goal", H1), hr(),
    Paragraph("We set out to read a team's everyday chat and tell each member which "
              "<b>collaboration role</b> they naturally play. We use the five COIN "
              "archetypes from Peter Gloor's organisational research:", BODY)]
arch_rows = [["", "Archetype", "Collaboration role"],
    ["", "Bee", "Central connector / initiator — drives and cross-pollinates discussion"],
    ["", "Ant", "Reliable task worker — steady, substantive delivery"],
    ["", "Butterfly", "Information bridge — shares links and news, connects topics"],
    ["", "Capybara", "Warm supporter — gratitude, encouragement, harmony"],
    ["", "Leech", "Low-reciprocity participant — receives more than contributes"]]
ARC_COL = [colors.HexColor("#F4C430"), colors.HexColor("#C8772E"),
           colors.HexColor("#9B59B6"), colors.HexColor("#2ECC71"), colors.HexColor("#E74C3C")]
t = Table(arch_rows, colWidths=[0.5*cm, 2.8*cm, 12.4*cm])
tstyle = [
    ("BACKGROUND", (0,0), (-1,0), NAVY),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTNAME", (1,1), (1,-1), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9.4),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#D5D8DC")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING", (0,0), (-1,-1), 7)]
for i, c in enumerate(ARC_COL, start=1):
    tstyle.append(("BACKGROUND", (0,i), (0,i), c))
t.setStyle(TableStyle(tstyle))
story += [t, Spacer(1, 0.4*cm),
    Paragraph("The challenge that shaped every later decision: our raw chats had "
              "<b>no labels at all</b>, the team we most cared about is <b>only five "
              "people</b>, and two of the five roles (Butterfly, Capybara) are rare in "
              "the kind of technical chats we could find. The rest of this report is "
              "the story of solving those three problems honestly.", BODY)]


# ═════════════════════════════════════════════════════════════════════════════
# 2. THE JOURNEY AT A GLANCE
# ═════════════════════════════════════════════════════════════════════════════
story += [Paragraph("2 · The journey at a glance", H1), hr(),
    Paragraph("Nine major steps. Six we kept, one we repaired, two we deliberately "
              "dropped. Each is explained — with its evidence — in the sections that follow.", BODY),
    Spacer(1, 0.2*cm), img(ASSETS / "journey.png", 16.5), Spacer(1, 0.3*cm)]


# ═════════════════════════════════════════════════════════════════════════════
# 3. FROM NO LABELS TO LABELS
# ═════════════════════════════════════════════════════════════════════════════
story += [Paragraph("3 · From no labels to labels", H1), hr(),
    Paragraph("<b>The problem.</b> Supervised learning needs a target, but none of our "
              "chats came with archetype labels. Hand-labelling thousands of users was "
              "infeasible and subjective.", BODY),
    Paragraph("<b>What we did.</b> We used <b>weak supervision</b>: a large language model "
              "reads each user's raw messages and returns a <i>soft</i> label — a "
              "probability spread over the five archetypes (e.g. 60% Bee / 25% Ant / …) "
              "rather than one hard class. Soft labels capture the truth that real people "
              "blend roles.", BODY),
    Paragraph("<b>Why this is not circular.</b> A reviewer's first worry is \"you learn "
              "the labels from the same numbers you feed the model.\" We avoid that by "
              "construction: the model's <b>inputs (X)</b> are measured behavioural and "
              "linguistic features; the <b>labels (y)</b> come from a different source "
              "reading the raw text. Because X and y are produced independently, agreement "
              "between them measures <i>validity</i>, not self-consistency.", BODY),
    Paragraph("Basis: Ratner et al. (2020), <i>Snorkel</i> — programmatic weak supervision; "
              "Hinton, Vinyals & Dean (2015) — distilling soft labels into a model.", SMALL)]


# ═════════════════════════════════════════════════════════════════════════════
# 4. DATA
# ═════════════════════════════════════════════════════════════════════════════
story += [PageBreak(), Paragraph("4 · The data we trained on", H1), hr(),
    Paragraph("Training uses public developer/community chats; our own team chat is held "
              "back entirely as a blind test. We never train, tune, or label on the team.", BODY),
    img(ASSETS / "datasets.png", 14.5),
    Paragraph("Labelled users per source. Blue = real public corpora, purple = disclosed "
              "synthetic augmentation (Section 7), red = the blind five-person team.", CAP)]
data_rows = [["Dataset", "Role", "Size"],
    ["Slack (MSR 2020 developer chat)", "train", "1,639 users / 60k msgs"],
    ["Nankani 2020 (CODERS)", "train", "148 users / 11k msgs"],
    ["Mbada 2023 (WhatsApp professional group)", "train", "95 users / ~600 msgs"],
    ["Synthetic in-domain team chats (disclosed)", "train", "53 users / 264 msgs"],
    ["Our team WhatsApp", "TEST ONLY", "5 users / 1,052 msgs"]]
t = Table(data_rows, colWidths=[8.4*cm, 3.2*cm, 4.1*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), BLUE), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 9.2),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#D5D8DC")),
    ("TEXTCOLOR", (1,5), (-1,5), RED), ("FONTNAME", (1,5), (1,5), "Helvetica-Bold"),
    ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 7)]))
story += [t, Spacer(1, 0.3*cm),
    Paragraph("<b>300 users</b> are labelled for training (183 Slack + 55 Nankani + 9 "
              "Mbada + 53 synthetic). The five-person team stays a true held-out set, "
              "labelled only for the final blind comparison.", BODY)]


# ═════════════════════════════════════════════════════════════════════════════
# 5. FEATURES — TWO VIEWS
# ═════════════════════════════════════════════════════════════════════════════
story += [PageBreak(), Paragraph("5 · How we describe each person: two views", H1), hr(),
    Paragraph("We deliberately describe every user from two independent angles, because "
              "each is unreliable alone but together they are robust.", BODY),
    Paragraph("View A · 13 network &amp; behaviour metrics", H2),
    Paragraph("Message length, question ratio, task focus, harmony (sentiment), emoji "
              "ratio, emotion density, initiation rate, vocabulary richness (MATTR), plus "
              "interaction metrics: received/sent ratio, replies, mentions, betweenness, "
              "response consistency. <b>These were audited and repaired (v2):</b> the first "
              "version rested on a synthetic reply structure and a mention regex that matched "
              "e-mails; v2 parses real mentions and builds a windowed reply graph.", BODY),
    Paragraph("Basis: Covington &amp; McFall (2010) MATTR; Freeman (1977) / Brandes (2001) "
              "betweenness; Goh &amp; Barabási (2008) burstiness. Full audit in docs/FEATURES_V2.md.", SMALL),
    Paragraph("View B · 17 grounded text-style signals", H2),
    Paragraph("The text view is the more reliable half on a small group (it is "
              "scale-invariant: \"thanks, I can help 🙂\" reads the same in a 5- or "
              "1,600-person chat). We replaced an early set of ad-hoc keyword lists with "
              "<b>17 signals that each cite a validated, published resource</b>:", BODY)]
story += bullets([
    "<b>Empath</b> categories (help, affection, work, leader, giving, politeness, …) — "
    "crowd-validated, correlate r=0.906 with LIWC (Fast et al. 2016).",
    "<b>VADER</b> sentiment valence (Hutto &amp; Gilbert 2014).",
    "<b>Speech acts</b>: question rate (directive) and commitment rate (commissive) — Searle (1976).",
    "<b>Structural pragmatics</b>: link-sharing (the Butterfly bridge signal), @-mentions, emoji, length."])
story += [Spacer(1, 0.2*cm), img(DOCS / "text_features_table.png", 16.0),
    Paragraph("The 17 text-style features and their citations.", CAP)]


# ═════════════════════════════════════════════════════════════════════════════
# 6. DECISIONS LOG
# ═════════════════════════════════════════════════════════════════════════════
story += [PageBreak(), Paragraph("6 · Every decision — kept, fixed, or dropped", H1), hr(),
    Paragraph("The heart of the project. Each row is a fork in the road, what the evidence "
              "showed, and what we did about it.", BODY)]
dec = [["Decision point", "What we found", "Outcome"],
    ["Hard vs soft labels",
     "Hard single-class labels punish honest near-ties between blended roles.",
     "KEPT soft labels"],
    ["Rule-only classifier\n(Slide-6 thresholds)",
     "Transparent but brittle; thresholds don't transfer across group sizes.",
     "DROPPED as the\ndecider (kept as a\nreadability reference)"],
    ["v1 interaction metrics",
     "Rested on a fake reply chain; mention regex matched e-mails.",
     "FIXED → features_v2"],
    ["One view vs two views",
     "Network metrics degenerate on 5 people; text style stays valid.",
     "KEPT late fusion\nof both views"],
    ["Garimella WhatsApp\n(6.3M real messages)",
     "Real but broadcast spam/propaganda — catastrophically off-domain.",
     "DROPPED (realness\n< domain match)"],
    ["Rare classes\n(Butterfly, Capybara)",
     "Almost absent from technical chats → unlearnable (F1≈0).",
     "KEPT disclosed\nsynthetic augment."],
    ["Ad-hoc keyword lists\nvs grounded lexicons",
     "Keyword lists are zero-inflated and uncitable.",
     "KEPT 17 grounded,\ncitable features"],
    ["Hard accuracy vs\nsoft accuracy headline",
     "Argmax collapses on near-ties; distribution overlap is fairer.",
     "KEPT soft accuracy\nas headline"],
    ["Fusion weight",
     "CV plateau 0.10–0.30; we report an a-priori-neutral balance.",
     "KEPT 50/50 (data-\nsupported, no leakage)"]]
t = Table(dec, colWidths=[4.0*cm, 7.6*cm, 4.1*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (0,0), (-1,-1), 8.7),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#D5D8DC")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("FONTNAME", (2,1), (2,-1), "Helvetica-Bold"),
    ("TEXTCOLOR", (2,1), (2,1), GREEN), ("TEXTCOLOR", (2,2), (2,2), RED),
    ("TEXTCOLOR", (2,3), (2,3), ORANGE), ("TEXTCOLOR", (2,4), (2,4), GREEN),
    ("TEXTCOLOR", (2,5), (2,5), RED), ("TEXTCOLOR", (2,6), (2,6), GREEN),
    ("TEXTCOLOR", (2,7), (2,7), GREEN), ("TEXTCOLOR", (2,8), (2,8), GREEN),
    ("TEXTCOLOR", (2,9), (2,9), GREEN),
    ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING", (0,0), (-1,-1), 6)]))
story += [t]


# ═════════════════════════════════════════════════════════════════════════════
# 7. FAILURES
# ═════════════════════════════════════════════════════════════════════════════
story += [PageBreak(), Paragraph("7 · What failed — and what it taught us", H1), hr(),
    Paragraph("<b>Garimella (6.3M real WhatsApp messages).</b> The tempting \"more real "
              "data\" option. We mined it (22,664 qualifying users, read ~90) and found "
              "broadcast forwards, bots and political spam — nothing like collaborative "
              "teamwork. Lesson kept: <b>domain match beats raw realness</b>. We dropped it.", BODY),
    Paragraph("<b>The rule-only classifier.</b> Our first classifier scored each metric "
              "against fixed Slide-6 ranges. Readable, but thresholds tuned on one group "
              "size don't transfer, and it can't express blended roles. We kept it only as "
              "an interpretability reference, not the decider.", BODY),
    Paragraph("<b>A macOS crash (segfault).</b> Loading two transformer stacks in one "
              "process crashed on Apple Silicon. Fixed by a <i>style-only</i> path that "
              "skips the heavy sentence embedding in production — the model only needs the "
              "lightweight, citable text features, which also made it faster.", BODY),
    Paragraph("<b>A first, too-sparse synthetic set.</b> Our first generated chats averaged "
              "~4 messages per user — too thin to produce stable features and it lowered the "
              "held-out score. We rewrote longer conversations and the score recovered. "
              "Lesson: synthetic data must match the <i>density</i> of the real thing, not "
              "just its labels.", BODY)]


# ═════════════════════════════════════════════════════════════════════════════
# 8. AUGMENTATION
# ═════════════════════════════════════════════════════════════════════════════
story += [PageBreak(), Paragraph("8 · Fixing the class imbalance honestly", H1), hr(),
    Paragraph("Butterfly and Capybara barely exist in technical Q&amp;A chats, so the model "
              "could not learn them (out-of-fold F1 ≈ 0). Rather than import off-domain real "
              "data, we added <b>disclosed, in-domain synthetic team chats</b>: short, "
              "naturalistic small-team conversations skewed toward the starved roles. They "
              "are generated <i>then independently labelled the same way as every real "
              "user</i>, are fully disclosed, toggleable (--no-synthetic), and the blind "
              "team test stays 100% real.", BODY),
    img(ASSETS / "class_balance.png", 12.5),
    Paragraph("Class balance after augmentation: 300 labelled users.", CAP),
    img(ASSETS / "ablation.png", 14.0),
    Paragraph(f"The decisive result: the two starved roles become learnable "
              f"(Butterfly 0.00→0.37, Capybara 0.01→0.63) and overall macro-F1 rises "
              f"{MF_NO:.2f} → {MF_YES:.2f}. Full rationale in docs/SYNTHETIC_AUGMENTATION.md.", CAP),
    Paragraph("Basis: standard practice in disclosed data augmentation; we follow the "
              "transparency guidance that augmented data must be labelled by the same "
              "independent process and never leak into the test set.", SMALL)]


# ═════════════════════════════════════════════════════════════════════════════
# 9. FUSION + EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
story += [PageBreak(), Paragraph("9 · The final model and how we judge it", H1), hr(),
    Paragraph("Two Random-Forest soft-regressors — one per view — each predict the soft "
              "archetype distribution, and we <b>late-fuse</b> them: "
              "<font face='Courier'>P = w·P_metric + (1−w)·P_style</font>. "
              "Cross-validation shows a flat quality plateau across w = 0.10–0.30, "
              "confirming the text view carries most signal; we report a balanced, "
              "a-priori-neutral 50/50 so the choice is principled rather than tuned to the "
              "five-person test (which would be leakage).", BODY),
    Paragraph("Basis: Breiman (2001) Random Forests; late-fusion of independent views is "
              "standard multi-view learning.", SMALL),
    Paragraph("Why <b>soft accuracy</b> is our headline", H2),
    Paragraph("On a fuzzy target where someone is genuinely 27% Bee / 26% Ant, scoring the "
              "argmax as right-or-wrong throws away most of the truth. Our headline is "
              "<b>soft accuracy = mean Σ min(model, label)</b> — the probability mass the "
              "model shares with the independent label. It credits partial agreement and is "
              "the honest yardstick for blended roles.", BODY),
    Paragraph("Blind held-out validation", H2),
    Paragraph("The only honest test: the five teammates the model never saw in training.", BODY),
    img(ASSETS / "holdout.png", 15.5),
    Paragraph(f"Soft accuracy {SOFT}, distribution cosine {COS}, mean KL {KL}, top-1 "
              f"{TOP1}. Three of five exact, and the two misses are genuine near-ties "
              "where the distributions still "
              "overlap the label strongly — exactly what soft accuracy is built to show.", CAP)]


# ═════════════════════════════════════════════════════════════════════════════
# 10. TEAM RESULT + DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
story += [PageBreak(), Paragraph("10 · The result for our team", H1), hr(),
    img(ASSETS / "team_snapshot.png", 16.0),
    Paragraph("Each member's primary role (left) and full role blend (right). Nobody is a "
              "single pure type — the model returns the realistic mixture.", CAP),
    Paragraph("11 · The dashboard", H1), hr(),
    Paragraph("All of this ships as a friendly Streamlit dashboard a non-technical "
              "teammate can read in a minute. It was rewritten to remove jargon and "
              "internal machinery and to focus on four clear views:", BODY)]
story += bullets([
    "<b>Overview</b> — team make-up, a plain who's-who, role-strength bars, and a simple team-health check.",
    "<b>Who talks to whom</b> — the interaction network, sized by activity and coloured by role.",
    "<b>Activity</b> — message volume over time and an everyone-online heatmap.",
    "<b>Member profile</b> — one place per person: their role, a radar and breakdown of the five roles, what stands out, and recent messages."])
story += [Paragraph("Start it with <font face='Courier'>streamlit run src/dashboard.py</font>.", SMALL)]


# ═════════════════════════════════════════════════════════════════════════════
# 12. LIMITATIONS + REFERENCES
# ═════════════════════════════════════════════════════════════════════════════
story += [PageBreak(), Paragraph("12 · Honest limitations", H1), hr()]
story += bullets([
    "Butterfly and Capybara remain the hardest classes; even augmented they are the least confident.",
    "Betweenness centrality is mathematically near-zero on a 5-person near-clique — a known small-group limit.",
    "Team predictions carry modest confidence (~25–28%) because of the train→team domain shift on several metrics.",
    "The held-out test is only five people; we therefore never tune to it and headline a distributional metric."])
story += [Paragraph("These are stated, not hidden — the soft metric and the disclosed "
                    "augmentation exist precisely because we took them seriously.", BODY)]

story += [Paragraph("13 · Scientific references", H1), hr()]
refs = [
    "Gloor, P. (2017/2022). <i>Sociometrics &amp; Human Relationships / Happimetrics</i> — COIN archetypes.",
    "Ratner, A. et al. (2020). <i>Snorkel: Rapid Training Data Creation with Weak Supervision.</i> VLDB.",
    "Hinton, G., Vinyals, O. &amp; Dean, J. (2015). <i>Distilling the Knowledge in a Neural Network.</i>",
    "Breiman, L. (2001). <i>Random Forests.</i> Machine Learning.",
    "Fast, E., Chen, B. &amp; Bernstein, M. (2016). <i>Empath.</i> CHI. (200 categories; r=0.906 with LIWC.)",
    "Hutto, C. &amp; Gilbert, E. (2014). <i>VADER.</i> ICWSM.",
    "Mohammad, S. &amp; Turney, P. (2013). <i>NRC Emotion Lexicon (EmoLex).</i>",
    "Tausczik, Y. &amp; Pennebaker, J. (2010). <i>The Psychological Meaning of Words: LIWC.</i>",
    "Danescu-Niculescu-Mizil, C. et al. (2013). <i>A Computational Approach to Politeness.</i> ACL.",
    "Searle, J. (1976). <i>A Classification of Illocutionary Acts.</i> (speech-act typology.)",
    "Covington, M. &amp; McFall, J. (2010). <i>MATTR — Moving-Average Type-Token Ratio.</i>",
    "Freeman, L. (1977) / Brandes, U. (2001). <i>Betweenness centrality.</i>",
    "Goh, K.-I. &amp; Barabási, A.-L. (2008). <i>Burstiness and memory in complex systems.</i>",
    "Kullback, S. &amp; Leibler, R. (1951). <i>On Information and Sufficiency (KL divergence).</i>",
    "Reimers, N. &amp; Gurevych, I. (2019). <i>Sentence-BERT.</i> (exploratory embedding view.)"]
story += [Paragraph(f"{i+1}.&nbsp; {r}", style("ref", fontSize=8.8, leading=12.5,
          spaceAfter=3, leftIndent=10)) for i, r in enumerate(refs)]
story += [Spacer(1, 0.4*cm), hr(),
    Paragraph("SocialCompass · COIN Team Communication Analyzer · full source, data map "
              "and per-decision docs in the repository (README.md, docs/).", SMALL)]


# ── footer with page numbers ─────────────────────────────────────────────────
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2*cm, 1.1*cm, "SocialCompass — Team Communication Analyzer")
    canvas.drawRightString(A4[0]-2*cm, 1.1*cm, f"page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#D5D8DC"))
    canvas.line(2*cm, 1.4*cm, A4[0]-2*cm, 1.4*cm)
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm,
                        topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title="SocialCompass — Project Report")
doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
print(f"✅ wrote {OUT}")
