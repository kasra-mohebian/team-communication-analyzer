"""
make_architecture_diagram.py — render the model architecture to a PNG/SVG for the
paper/slides. Pure matplotlib (no graphviz dependency).

    python src/make_architecture_diagram.py  →  archive/architecture-historical.png (+ .svg)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import config

C = {
    "raw":   "#34495E", "x": "#2E86C1", "y": "#27AE60",
    "model": "#8E44AD", "fuse": "#E67E22", "out": "#16A085", "note": "#7F8C8D",
}


def box(ax, x, y, w, h, text, color, fs=10, fc=None, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.5, edgecolor=color,
                                facecolor=fc if fc else color, alpha=1.0 if fc is None else 1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold", wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#555", style="-|>", lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, lw=lw, color=color, ls=ls,
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(12.5, 9))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 9); ax.axis("off")

    ax.text(6.25, 8.72, "SocialCompass — COIN Archetype Classifier (late-fusion, weak-supervised)",
            ha="center", fontsize=13.5, weight="bold", color="#222")

    # raw data
    box(ax, 0.6, 7.5, 7.1, 0.78,
        "RAW MESSAGES   —   training: Slack · Nankani · Mbada · Synthetic",
        C["raw"], fs=10.5)
    box(ax, 8.0, 7.5, 3.9, 0.78,
        "TEAM WhatsApp\n(TEST ONLY — never trained on)", C["raw"], fs=9.5)

    # X paths split
    ax.text(2.6, 7.05, "INPUTS  X  (measured signals)", ha="center", fontsize=9.5,
            color=C["x"], weight="bold")
    ax.text(9.6, 7.05, "LABELS  y  (independent)", ha="center", fontsize=9.5,
            color=C["y"], weight="bold")

    # two feature views
    box(ax, 0.6, 5.55, 3.25, 1.25,
        "VIEW A · 13 METRICS\nfeatures.py\nbehaviour + network + ML\n(MATTR, betweenness,\nreplies, mentions …)",
        C["x"], fs=8.6, fc="#EAF2FB", tc="#1B4F72")
    box(ax, 4.15, 5.55, 3.55, 1.25,
        "VIEW B · 17 TEXT-STYLE SIGNALS\ntext_features.py\nEmpath + VADER + speech-acts\n(help, affection, url-share,\nquestions, commitments …)",
        C["x"], fs=8.6, fc="#EAF2FB", tc="#1B4F72")

    # Opus labeler
    box(ax, 8.0, 5.55, 3.9, 1.25,
        "LLM reads RAW TEXT\n→ soft archetype label\n(no metrics seen →\nnot circular)",
        C["y"], fs=9, fc="#E8F8F0", tc="#145A32")

    arrow(ax, 3.0, 7.5, 2.2, 6.82, C["x"])
    arrow(ax, 4.6, 7.5, 5.9, 6.82, C["x"])
    arrow(ax, 9.8, 7.5, 9.9, 6.82, C["y"])

    # two RF sub-models
    box(ax, 0.9, 3.95, 2.65, 1.0, "Random Forest A\nmetrics → P_metric", C["model"], fs=9)
    box(ax, 4.45, 3.95, 2.95, 1.0, "Random Forest B\nstyle → P_style", C["model"], fs=9)

    arrow(ax, 2.2, 5.55, 2.2, 4.95, C["x"])
    arrow(ax, 5.9, 5.55, 5.9, 4.95, C["x"])

    # labels feed BOTH models (supervision)
    arrow(ax, 9.9, 5.55, 6.0, 4.55, C["y"], ls="--", lw=1.6)
    arrow(ax, 9.9, 5.55, 3.0, 4.35, C["y"], ls="--", lw=1.6)
    ax.text(7.2, 4.78, "supervises both (distillation)", fontsize=8, color=C["y"],
            style="italic", rotation=0)

    # fusion
    box(ax, 2.5, 2.35, 3.4, 1.0,
        "LATE FUSION\nP = w·P_metric + (1−w)·P_style\nw = view balance (CV / a-priori)",
        C["fuse"], fs=9)
    arrow(ax, 2.2, 3.95, 3.4, 3.35, C["model"])
    arrow(ax, 5.9, 3.95, 5.0, 3.35, C["model"])

    # output
    box(ax, 2.2, 0.7, 4.0, 1.0,
        "SOFT distribution over 5 archetypes\nBee · Ant · Butterfly · Capybara · Leech",
        C["out"], fs=9)
    arrow(ax, 4.2, 2.35, 4.2, 1.7, C["fuse"])

    # held-out validation (team) — right column
    box(ax, 8.0, 2.35, 3.9, 2.2,
        "BLIND HELD-OUT VALIDATION\n\nteam metrics + team text\n→ model prediction\nvs independent labels\n\nHEADLINE: soft accuracy\n(mass overlap, not argmax)",
        C["out"], fs=8.8, fc="#E8F6F3", tc="#0E6655")
    arrow(ax, 9.9, 7.5, 9.9, 4.55, C["raw"], ls=":", lw=1.4)
    ax.text(10.55, 6.0, "team\n(test)", fontsize=7.5, color=C["raw"], style="italic")
    arrow(ax, 6.2, 1.2, 8.0, 2.7, C["out"], ls="--", lw=1.5)

    ax.text(6.25, 0.18,
            "X (measured features) and y (independent text judgments) come from independent sources → accuracy measures validity, not self-consistency.",
            ha="center", fontsize=8.2, color=C["note"], style="italic")

    out_png = str(config.ROOT / "docs" / "architecture.png")
    out_svg = str(config.ROOT / "docs" / "architecture.svg")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print(f"✅ wrote {out_png}\n✅ wrote {out_svg}")


if __name__ == "__main__":
    main()
