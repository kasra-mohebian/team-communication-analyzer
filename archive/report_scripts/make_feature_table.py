"""
make_feature_table.py — render the 17 grounded text-style features as a citable
table image + PDF for the paper/slides.

    python src/make_feature_table.py  →  docs/text_features_table.png (+ .pdf)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

ARCH_COLOR = {"Bee": "#F1C40F", "Ant": "#7F8C8D", "Butterfly": "#9B59B6",
              "Capybara": "#27AE60", "Affect": "#E67E22", "Contrast": "#BDC3C7"}

# (feature, measures, archetype, source)
ROWS = [
    ("emp_help", "messages expressing helping", "Capybara", "Empath (Fast 2016)"),
    ("emp_affection", "warmth / care language", "Capybara", "Empath (Fast 2016)"),
    ("emp_positive_emotion", "positive affect", "Capybara", "Empath (Fast 2016)"),
    ("emp_negative_emotion", "negative affect", "Contrast", "Empath (Fast 2016)"),
    ("emp_work", "task / work language", "Ant", "Empath (Fast 2016)"),
    ("emp_achievement", "completion / achievement", "Ant", "Empath (Fast 2016)"),
    ("emp_communication", "communicative / social verbs", "Bee", "Empath (Fast 2016)"),
    ("emp_leader", "leadership / directing", "Bee", "Empath (Fast 2016)"),
    ("emp_giving", "sharing / giving", "Butterfly", "Empath (Fast 2016)"),
    ("emp_politeness", "polite forms", "Capybara", "Empath; Danescu (2013)"),
    ("vader_compound", "mean sentiment valence", "Capybara", "VADER (Hutto 2014)"),
    ("q_rate", "question rate (directive act)", "Bee", "Searle (1976)"),
    ("mention_rate", "@-addressing others", "Bee", "structural pragmatics"),
    ("url_rate", "link / resource sharing", "Butterfly", "structural (bridge signal)"),
    ("emoji_rate", "emoji per message", "Affect", "structural pragmatics"),
    ("commit_rate", "commitments ('I'll / I can')", "Ant", "commissive act, Searle (1976)"),
    ("avg_len_norm", "mean message length (/20)", "Ant", "structural pragmatics"),
]

REFS = ("Empath: Fast, Chen & Bernstein (2016) CHI — 200 categories, r=0.906 with LIWC   |   "
        "VADER: Hutto & Gilbert (2014) ICWSM   |   LIWC: Tausczik & Pennebaker (2010)\n"
        "Politeness: Danescu-Niculescu-Mizil et al. (2013) ACL   |   Speech acts: Searle (1976)   |   "
        "NRC EmoLex: Mohammad & Turney (2013)   |   COIN archetypes: Gloor (2017)")


def main():
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    ax.axis("off")
    ax.set_title("Text-style features — grounded & citable (17 signals, production text view)",
                 fontsize=14, weight="bold", pad=16)

    headers = ["#", "Feature", "What it measures", "Archetype", "Scientific source"]
    cell_text = [[str(i + 1), r[0], r[1], r[2], r[3]] for i, r in enumerate(ROWS)]

    tbl = ax.table(cellText=cell_text, colLabels=headers, loc="center",
                   cellLoc="left", colWidths=[0.04, 0.18, 0.27, 0.13, 0.30])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)
    tbl.scale(1, 1.5)

    # header styling
    for c in range(len(headers)):
        cell = tbl[0, c]
        cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", weight="bold")

    # row striping + archetype color chip on the Archetype column
    for i, r in enumerate(ROWS, start=1):
        bg = "#F7F9FB" if i % 2 else "#FFFFFF"
        for c in range(len(headers)):
            tbl[i, c].set_facecolor(bg)
            tbl[i, c].set_edgecolor("#D5DBDB")
        ac = tbl[i, 3]
        ac.set_facecolor(ARCH_COLOR.get(r[2], "#FFFFFF"))
        ac.set_text_props(color="white" if r[2] != "Contrast" else "#555", weight="bold")
        tbl[i, 1].set_text_props(family="monospace", weight="bold", color="#1B4F72")

    fig.text(0.5, 0.045, REFS, ha="center", fontsize=7.6, color="#566573")
    fig.text(0.5, 0.012,
             "All values are per-user rates/means → scale-invariant. X (features) and y (Opus text labels) are independent → not circular.",
             ha="center", fontsize=7.8, color="#566573", style="italic")

    png = str(config.ROOT / "docs" / "text_features_table.png")
    pdf = str(config.ROOT / "docs" / "text_features_table.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"✅ wrote {png}\n✅ wrote {pdf}")


if __name__ == "__main__":
    main()
