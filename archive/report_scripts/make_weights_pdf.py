"""
make_weights_pdf.py — explain WHAT THE MODEL WEIGHS, for teammates.

Produces:
  • docs/assets/weights_global.png   — effective importance of every feature
  • docs/assets/weights_perarch.png  — feature × archetype heatmap (permutation)
  • outputs/results/feature_weights.json        (global table)
  • outputs/results/feature_weights_perarch.json(per-archetype matrix)
  • docs/FEATURE_WEIGHTS.pdf          — concise English explainer + both tables

The global table is the model's Gini importance scaled by the 0.5/0.5 fusion
weight (a feature's share of the final decision). The per-archetype matrix is
PERMUTATION importance per output: shuffle one feature, measure how much each
archetype's predicted probability degrades — i.e. which feature drives which
role, learned from data (not a hand-written rule table).

    python src/make_weights_pdf.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
# importing these into __main__ lets joblib unpickle the saved FusionModel
from train_ml import (                                    # noqa: F401
    FusionModel, SoftRegressor, MeanBaseline, LogRegBaseline,
    load_training_matrix, load_style_features, ARCH_KEYS,
)
import joblib

ARCH = ["Bee", "Ant", "Butterfly", "Capybara", "Leech"]   # display order
COL  = {"Bee": "#F4C430", "Ant": "#C8772E", "Butterfly": "#9B59B6",
        "Capybara": "#2ECC71", "Leech": "#E74C3C"}
ASSETS = config.ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
PDF = config.ROOT / "docs" / "FEATURE_WEIGHTS.pdf"
plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold",
                     "figure.facecolor": "white"})


# ════════════════════════════════════════════════════════════════════════════
# 1. Load model + data
# ════════════════════════════════════════════════════════════════════════════
def load():
    bundle = joblib.load(config.ML_MODEL)
    model = bundle["model"]
    w = float(bundle["fusion_metric_w"])
    mnames = list(bundle["metrics"])
    snames = list(bundle["style_names"])
    merged, X, y, sw, hard, _ = load_training_matrix(drop_synthetic=False)
    S = load_style_features(merged)
    return model, w, mnames, snames, X, S, y


# ════════════════════════════════════════════════════════════════════════════
# 2. Global effective importance  (Gini × fusion weight)
# ════════════════════════════════════════════════════════════════════════════
def global_table(model, w, mnames, snames):
    mi, _ = model.metric_.feature_importances()
    si, _ = model.style_.feature_importances()
    rows = [("metric", n, float(v), float(w * v)) for n, v in zip(mnames, mi)]
    rows += [("text-style", n, float(v), float((1 - w) * v)) for n, v in zip(snames, si)]
    rows.sort(key=lambda r: r[3], reverse=True)
    json.dump(
        {"fusion_metric_w": w,
         "note": "effective_weight = view_weight * within-view Gini importance; rows sum to 1.0",
         "effective_importance": [
             {"view": v, "feature": n, "in_view_importance": round(iv, 4),
              "effective_weight": round(e, 4)} for v, n, iv, e in rows]},
        open(config.RESULTS / "feature_weights.json", "w"), indent=2)
    return rows


# ════════════════════════════════════════════════════════════════════════════
# 3. Per-archetype PERMUTATION importance
#    shuffle feature j → rise in that archetype's prediction error (MSE)
# ════════════════════════════════════════════════════════════════════════════
def perarch_matrix(model, mnames, snames, X, S, y, n_repeats=20, seed=0):
    rng = np.random.default_rng(seed)
    base = model.predict(X, S)
    base_mse = ((y - base) ** 2).mean(axis=0)                  # 5

    names, mat = [], []
    for j, n in enumerate(mnames):                             # permute metric cols
        d = np.zeros(5)
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            d += ((y - model.predict(Xp, S)) ** 2).mean(axis=0) - base_mse
        names.append(n); mat.append(d / n_repeats)
    for j, n in enumerate(snames):                             # permute style cols
        d = np.zeros(5)
        for _ in range(n_repeats):
            Sp = S.copy(); Sp[:, j] = rng.permutation(Sp[:, j])
            d += ((y - model.predict(X, Sp)) ** 2).mean(axis=0) - base_mse
        names.append(n); mat.append(d / n_repeats)

    mat = np.clip(np.array(mat), 0, None)                      # drop noise < 0
    # normalize each archetype column to a share (sums to 1 per archetype)
    col = mat / np.maximum(mat.sum(axis=0, keepdims=True), 1e-12)
    json.dump(
        {"note": "permutation importance per archetype (share per column, sums to 1)",
         "features": names,
         "archetypes": ARCH_KEYS,
         "share": {ARCH_KEYS[k]: {names[i]: round(float(col[i, k]), 4)
                                  for i in range(len(names))} for k in range(5)}},
        open(config.RESULTS / "feature_weights_perarch.json", "w"), indent=2)
    return names, col                                         # col: feat × 5 (ARCH_KEYS order)


# ════════════════════════════════════════════════════════════════════════════
# 4. Charts
# ════════════════════════════════════════════════════════════════════════════
def chart_global(rows):
    rows = rows[::-1]                                          # smallest at bottom
    labels = [f"{n}" for _, n, _, _ in rows]
    vals = [e * 100 for _, _, _, e in rows]
    cols = ["#3498DB" if v == "metric" else "#9B59B6" for v, _, _, _ in rows]
    fig, ax = plt.subplots(figsize=(8.6, 8.8))
    ax.barh(labels, vals, color=cols, alpha=.92, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(v + 0.05, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("share of the final decision  (%)")
    ax.set_title("How much the model leans on each feature\n"
                 "blue = network-metric view · purple = text-style view", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "weights_global.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ weights_global.png")


def chart_perarch(names, col, top=12):
    # show the union of each archetype's top features for a compact heatmap
    order = np.argsort(col.max(axis=1))[::-1][:top]
    sub = col[order]
    labels = [names[i] for i in order]
    # reorder columns to display order (ARCH) from ARCH_KEYS
    key_to_disp = [ARCH_KEYS.index(a.lower()) for a in ARCH]
    sub = sub[:, key_to_disp]
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    im = ax.imshow(sub, cmap="magma_r", aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(ARCH, rotation=20, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for k in range(5):
            v = sub[i, k]
            ax.text(k, i, f"{v*100:.0f}", ha="center", va="center",
                    fontsize=8, color="white" if v > sub.max()*0.55 else "#222")
    ax.set_title("Which feature drives which archetype\n"
                 "(permutation importance · % share within each archetype)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="share")
    fig.tight_layout()
    fig.savefig(ASSETS / "weights_perarch.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ weights_perarch.png")
    return names, col


# ════════════════════════════════════════════════════════════════════════════
# 5. PDF
# ════════════════════════════════════════════════════════════════════════════
def build_pdf(rows, names, col):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable)

    ss = getSampleStyleSheet()
    NAVY = colors.HexColor("#1B2A4A")
    BLUE = colors.HexColor("#3498DB")

    def st(name, **kw):
        kw.setdefault("parent", ss["Normal"])
        return ParagraphStyle(name, **kw)

    h1 = st("h1", fontName="Helvetica-Bold", fontSize=18, textColor=NAVY, spaceAfter=4, leading=22)
    h2 = st("h2", fontName="Helvetica-Bold", fontSize=12.5, textColor=BLUE, spaceBefore=12, spaceAfter=5)
    body = st("body", fontSize=9.7, leading=14, spaceAfter=6)
    small = st("small", fontSize=8.4, leading=11.5, textColor=colors.HexColor("#555"))

    doc = SimpleDocTemplate(str(PDF), pagesize=A4,
                            leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=1.6*cm, bottomMargin=1.5*cm,
                            title="SocialCompass — What the model weighs")
    E = []
    E.append(Paragraph("SocialCompass — What the Model Weighs", h1))
    E.append(Paragraph("A plain-English look at how our archetype classifier makes its decision, "
                       "and which signals matter for each role.", small))
    E.append(HRFlowable(width="100%", color=BLUE, thickness=1.2, spaceBefore=6, spaceAfter=8))

    E.append(Paragraph("In one paragraph", h2))
    E.append(Paragraph(
        "The classifier is two <b>Random Forests</b> working side by side: a "
        "<b>network-metric view</b> (13 behaviour/graph signals) and a "
        "<b>text-style view</b> (17 language signals). Their predictions are averaged "
        "<b>50 / 50</b>. The output is a <b>probability spread across all five archetypes</b> "
        "&mdash; not a single hard label &mdash; so mixed, in-between profiles are expressed "
        "naturally. Random Forests do not have a single linear weight per feature; instead each "
        "feature has an <b>importance</b>: how much the model relies on it. The two tables below "
        "show (1) the overall importance of every feature, and (2) which features drive each "
        "individual archetype.", body))

    E.append(Paragraph("Three ideas to read the tables", h2))
    for b in [
        "<b>Importance ≈ weight.</b> It measures how much a feature is used to separate the "
        "classes. It is a <i>magnitude</i> only &mdash; it tells you a feature matters, not the "
        "direction (high or low) that points to a given role.",
        "<b>50/50 fusion.</b> Because the two views are averaged equally, a feature's "
        "<i>effective weight</i> in the final decision is 0.5 × its importance inside its own "
        "view. All 30 features together add up to 100%.",
        "<b>Per-archetype = permutation.</b> We shuffle one feature and watch whose prediction "
        "breaks the most. The role that degrades most is the one that feature drives. This is "
        "<b>learned from data</b> &mdash; unlike an earlier hand-written rule table we retired.",
    ]:
        E.append(Paragraph("• " + b, body))

    # ── global table figure ──
    E.append(Paragraph("1 · Overall feature weighting", h2))
    E.append(Paragraph(
        "Top signals are stylistic on both sides &mdash; emoji use, link sharing, and emotional "
        "tone &mdash; which matches our design choice to let the scale-invariant text view carry "
        "most of the decision on a small team.", body))
    E.append(Image(str(ASSETS / "weights_global.png"), width=15.0*cm, height=15.3*cm))

    # ── per-archetype figure ──
    E.append(Paragraph("2 · Which feature drives which archetype", h2))
    # top-3 per archetype as a quick table
    key_idx = {a: ARCH_KEYS.index(a.lower()) for a in ARCH}
    head = ["Archetype", "Top signals the model uses (in order)"]
    data = [head]
    for a in ARCH:
        k = key_idx[a]
        order = np.argsort(col[:, k])[::-1][:3]
        sig = ", ".join(f"{names[i]} ({col[i,k]*100:.0f}%)" for i in order)
        data.append([a, sig])
    t = Table(data, colWidths=[3.0*cm, 12.0*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FB")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCD6E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    for i, a in enumerate(ARCH, 1):
        t.setStyle(TableStyle([("TEXTCOLOR", (0, i), (0, i), COL[a])]))
    E.append(t)
    E.append(Spacer(1, 8))
    E.append(Image(str(ASSETS / "weights_perarch.png"), width=12.6*cm, height=11.2*cm))

    E.append(Paragraph("A caveat we keep honest", h2))
    E.append(Paragraph(
        "<font color='#444'>betweenness_centrality</font> looks useful during training on large "
        "communities, but it is mathematically near-zero for everyone in a small, fully-connected "
        "5-person chat &mdash; so its real effect on our own team prediction is minimal. This is "
        "exactly why the text-style view shares the decision 50/50 rather than trusting the "
        "network metrics alone.", small))

    doc.build(E)
    print(f"  ✓ {PDF.name}  ->  {PDF}")


# ════════════════════════════════════════════════════════════════════════════
def main():
    print("\n📊  Building feature-weight explainer …")
    model, w, mnames, snames, X, S, y = load()
    rows = global_table(model, w, mnames, snames)
    names, col = perarch_matrix(model, mnames, snames, X, S, y)
    chart_global(rows)
    chart_perarch(names, col)
    build_pdf(rows, names, col)
    print("  done.\n")


if __name__ == "__main__":
    main()
