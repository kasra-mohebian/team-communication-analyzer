"""
make_report_assets.py — render every chart used by the PDF report and the slide
deck into docs/assets/*.png. Pure matplotlib; numbers read from the result JSONs
so the figures always match the production model.

    python src/make_report_assets.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config
from pathlib import Path

ASSETS = config.ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

ARCH = ["Bee", "Ant", "Butterfly", "Capybara", "Leech"]
COL  = {"Bee": "#F4C430", "Ant": "#C8772E", "Butterfly": "#9B59B6",
        "Capybara": "#2ECC71", "Leech": "#E74C3C"}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#cccccc",
                     "axes.titleweight": "bold", "figure.facecolor": "white"})


def save(fig, name):
    p = ASSETS / name
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  ✓", p.name)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset composition (labeled users per source)
# ─────────────────────────────────────────────────────────────────────────────
def fig_datasets():
    sources = ["Slack\n(dev chat)", "Nankani\n(CODERS)", "Mbada\n(WhatsApp prof.)",
               "Synthetic\n(disclosed)", "Team\n(test only)"]
    labeled = [183, 55, 9, 53, 5]
    colors  = ["#3498DB", "#3498DB", "#3498DB", "#9B59B6", "#E74C3C"]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.bar(sources, labeled, color=colors, alpha=.92, edgecolor="white")
    for b, v in zip(bars, labeled):
        ax.text(b.get_x()+b.get_width()/2, v+2, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("labelled users")
    ax.set_title("Labelled users per data source  ·  300 for training + 5 held-out team")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.99, 0.95, "blue = real training • purple = disclosed synthetic • red = blind test",
            transform=ax.transAxes, ha="right", fontsize=8.5, color="#777")
    save(fig, "datasets.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Class balance of the 300 training labels
# ─────────────────────────────────────────────────────────────────────────────
def fig_class_balance():
    bal = {"Bee": 104, "Ant": 73, "Leech": 56, "Capybara": 35, "Butterfly": 32}
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bars = ax.bar(list(bal.keys()), list(bal.values()),
                  color=[COL[k] for k in bal], alpha=.92, edgecolor="white")
    for b, v in zip(bars, bal.values()):
        ax.text(b.get_x()+b.get_width()/2, v+1.2, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("users")
    ax.set_title("Class balance after augmentation (300 labelled users)")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "class_balance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Synthetic ablation — per-class F1 with vs without augmentation
# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation():
    ab = json.load(open(config.RESULTS / "synthetic_ablation.json"))
    wo = ab["oof_cv"]["without_synthetic"]["per_class_f1"]
    wi = ab["oof_cv"]["with_synthetic"]["per_class_f1"]
    order = ["bee", "ant", "butterfly", "capybara", "leech"]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.bar(x-w/2, [wo[k] for k in order], w, label="without synthetic", color="#bdc3c7")
    ax.bar(x+w/2, [wi[k] for k in order], w, label="with synthetic",
           color=[COL[k.capitalize()] for k in order], alpha=.95)
    ax.set_xticks(x); ax.set_xticklabels([k.capitalize() for k in order])
    ax.set_ylabel("out-of-fold F1"); ax.set_ylim(0, 0.75)
    ax.set_title("Disclosed augmentation makes the rare roles learnable\n"
                 f"macro-F1 {ab['oof_cv']['without_synthetic']['macro_f1']:.2f} → "
                 f"{ab['oof_cv']['with_synthetic']['macro_f1']:.2f}")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    for xi, k in zip(x, order):
        ax.text(xi+w/2, wi[k]+0.01, f"{wi[k]:.2f}", ha="center", fontsize=8.5, fontweight="bold")
    save(fig, "ablation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Blind held-out validation (current production model)
# ─────────────────────────────────────────────────────────────────────────────
def fig_holdout():
    ho = json.load(open(config.HOLDOUT_COMPARISON))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    # left: headline metrics
    ax = axes[0]; ax.axis("off")
    metrics = [("Soft accuracy", f"{ho['soft_accuracy']*100:.0f}%", "#2ECC71"),
               ("Distribution cosine", f"{ho['mean_cosine']:.2f}", "#3498DB"),
               ("Top-1 agreement", f"{int(round(ho['top1_agreement']*5))}/5", "#F4C430"),
               ("Mean KL (lower=better)", f"{ho['mean_kl']:.2f}", "#9B59B6")]
    for i, (lab, val, c) in enumerate(metrics):
        y = 0.82 - i*0.24
        ax.add_patch(plt.Rectangle((0.04, y-0.09), 0.92, 0.18, transform=ax.transAxes,
                                   facecolor=c, alpha=.14, edgecolor=c, lw=1.4))
        ax.text(0.10, y, val, transform=ax.transAxes, fontsize=20, fontweight="bold",
                va="center", color=c)
        ax.text(0.42, y, lab, transform=ax.transAxes, fontsize=11, va="center")
    ax.text(0.04, 1.02, "Blind held-out team — never trained on", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

    # right: per-person soft accuracy
    ax = axes[1]
    pp = pd.DataFrame(ho["per_person"])
    pp = pp.sort_values("soft_accuracy")
    bars = ax.barh(pp["author"], pp["soft_accuracy"]*100,
                   color=["#2ECC71" if m else "#E67E22" for m in pp["match"]], alpha=.9)
    for b, (_, r) in zip(bars, pp.iterrows()):
        tag = "✓" if r["match"] else "≈"
        ax.text(r["soft_accuracy"]*100+0.6, b.get_y()+b.get_height()/2,
                f"{tag} {r['model']}", va="center", fontsize=9)
    ax.set_xlim(0, 100); ax.set_xlabel("soft accuracy (%)")
    ax.set_title("Per-member  (green = top-1 hit, orange = near-tie)", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(wspace=0.25, top=0.86)
    save(fig, "holdout.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. The decision journey (timeline of what we kept / dropped)
# ─────────────────────────────────────────────────────────────────────────────
def fig_journey():
    steps = [
        ("Unlabelled\nchats", "raw Slack / Nankani /\nWhatsApp — no labels", "#34495E", "keep"),
        ("Independent\nlabels", "read raw text →\nsoft labels", "#27AE60", "keep"),
        ("Network\nmetrics", "first feature\nview (v1)", "#2E86C1", "fixed→v2"),
        ("Rule-only\nclassifier", "Slide-6\nthresholds", "#7F8C8D", "drop"),
        ("Two-view\nfusion", "metrics +\ntext style", "#E67E22", "keep"),
        ("Garimella\ndata", "6.3M msgs,\noff-domain", "#C0392B", "drop"),
        ("Synthetic\naugment.", "rare roles\nlearnable", "#9B59B6", "keep"),
        ("Grounded\ntext feats", "Empath +\nVADER + acts", "#16A085", "keep"),
        ("Soft\naccuracy", "honest\nheadline", "#2ECC71", "keep"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 4.2)); ax.axis("off")
    n = len(steps); xw = 1.0/n
    for i, (title, sub, c, tag) in enumerate(steps):
        x = i*xw + 0.008
        kept = tag != "drop"
        ax.add_patch(plt.Rectangle((x, 0.40), xw-0.05, 0.34, transform=ax.transAxes,
                     facecolor=c, alpha=.92 if kept else .25, edgecolor=c, lw=1.5))
        ax.text(x+(xw-0.05)/2, 0.57, title, transform=ax.transAxes, ha="center",
                va="center", fontsize=7.7, fontweight="bold", linespacing=1.1,
                color="white" if kept else "#777")
        ax.text(x+(xw-0.05)/2, 0.32, sub, transform=ax.transAxes, ha="center",
                va="top", fontsize=6.8, color="#444", linespacing=1.1)
        badge = {"keep": ("KEPT", "#27AE60"), "drop": ("DROPPED", "#C0392B"),
                 "fixed→v2": ("FIXED", "#E67E22")}[tag]
        ax.text(x+(xw-0.05)/2, 0.80, badge[0], transform=ax.transAxes, ha="center",
                fontsize=6.6, fontweight="bold", color=badge[1])
        if i < n-1:
            ax.annotate("", xy=(x+xw-0.002, 0.57), xytext=(x+xw-0.05, 0.57),
                        xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.3))
    ax.set_title("Project journey — every step, and whether it survived into the final model",
                 fontsize=12, pad=4)
    save(fig, "journey.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Team result snapshot (the dashboard's core output, as a static figure)
# ─────────────────────────────────────────────────────────────────────────────
def fig_team_snapshot():
    res = pd.read_json(config.WA_RESULTS, lines=True)
    fitcols = ["bee_pct", "ant_pct", "butterfly_pct", "capybara_pct", "leech_pct"]
    fig, (axp, axb) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1.4]})
    # pie of primary roles
    prim = res["archetype"].str.split(" ", n=1).str[1].value_counts()
    axp.pie(prim.values, labels=prim.index, autopct="%1.0f%%",
            colors=[COL.get(k, "#999") for k in prim.index], startangle=140,
            wedgeprops=dict(width=0.42, edgecolor="white"),
            textprops=dict(fontsize=10))
    axp.set_title("Primary role per member")
    # stacked role strength
    res2 = res.sort_values("bee_pct")
    bottom = np.zeros(len(res2))
    for col, name in zip(fitcols, ARCH):
        axb.barh(res2["author"], res2[col], left=bottom, color=COL[name],
                 label=name, alpha=.92)
        bottom += res2[col].values
    axb.set_xlabel("role strength (%)"); axb.set_xlim(0, 100)
    axb.set_title("Every member is a blend of roles")
    axb.legend(ncol=5, fontsize=8, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.12))
    axb.spines[["top", "right"]].set_visible(False)
    save(fig, "team_snapshot.png")


if __name__ == "__main__":
    print("Rendering report assets → docs/assets/")
    fig_datasets()
    fig_class_balance()
    fig_ablation()
    fig_holdout()
    fig_journey()
    fig_team_snapshot()
    print("done.")
