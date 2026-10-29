"""
classify_archetypes.py
══════════════════════
RULE-BASED layer of the hybrid COIN archetype classifier.

Pipeline:
    raw 13 metrics (features.py)
      → 0-4 score per metric
            • 10 metrics  : ABSOLUTE thresholds (Slides 2 & 4)
            • 3  metrics  : RANK-normalized percentile → 0-4 (README_metrics.pdf)
      → for each person × each archetype: +1 if the 0-4 score falls inside the
        archetype's Slide-6 expected range, else 0
      → raw % = matched metrics / 13
      → normalize the 5 archetype %s so they sum to 100%
      → hard label = highest-scoring archetype

Run directly  →  scores `whatsapp_clean.json`, writes `wa_rule_results.json`.
Imported by   →  `train_and_predict.py` (uses score_features / classify / SLIDE6).

Every threshold and modeling choice cites its source inline.
"""

import math
import numpy as np
import pandas as pd

from features import (
    ABSOLUTE_METRICS, RANK_METRICS, ALL_METRICS,
    extract_features, sanity_check_features, load_clean,
)

ARCHETYPES = ["🐝 Bee", "🐜 Ant", "🦋 Butterfly", "🦫 Capybara", "🔴 Leech"]

# ════════════════════════════════════════════════════════════════════════════
# ABSOLUTE 0-4 SCORING BANDS  (Statusmeeting3_team5.pptx, Slides 2 & 4)
# Each entry is a list of upper-inclusive cut points → bands [0,1,2,3,4].
# A value v scores the index of the first band whose upper bound it does not
# exceed; values above the last cut point score 4.
# ════════════════════════════════════════════════════════════════════════════
ABSOLUTE_BANDS = {
    # avg_msg_length (words):  <3→0 · 3-5→1 · 6-19→2 · 20-30→3 · >30→4
    "avg_msg_length":   [2.999, 5, 19, 30],
    # replies_sent (rate):     0→0 · .01-.24→1 · .25-.49→2 · .50-.74→3 · .75-1→4
    "replies_sent":     [0.0, 0.24, 0.49, 0.74],
    # question_ratio:          0-.2→0 · .21-.4→1 · .41-.6→2 · .61-.8→3 · .81-1→4
    "question_ratio":   [0.20, 0.40, 0.60, 0.80],
    # avg_mentions:            0→0 · .01-.20→1 · .21-.50→2 · .51-1.0→3 · >1.0→4
    "avg_mentions":     [0.0, 0.20, 0.50, 1.00],
    # task_focus_score:        0-.10→0 · .11-.29→1 · .3-.6→2 · .61-.80→3 · .81-1→4
    "task_focus_score": [0.10, 0.29, 0.60, 0.80],
    # emoji_ratio:             0→0 · .01-.1→1 · .11-.29→2 · .3-.6→3 · .61-1→4
    "emoji_ratio":      [0.0, 0.10, 0.29, 0.60],
    # emotion_density:         0-.24→0 · .25-.49→1 · .5-.74→2 · .75-.89→3 · .9-1→4
    "emotion_density":  [0.24, 0.49, 0.74, 0.89],
    # initiation_rate:         0→0 · .01-.24→1 · .25-.49→2 · .5-.74→3 · .75-1→4
    "initiation_rate":  [0.0, 0.24, 0.49, 0.74],
    # harmony_score:           no band on the deck → standard quintiles
    # (project decision #3). 0-.2→0 · .2-.4→1 · .4-.6→2 · .6-.8→3 · .8-1→4
    "harmony_score":    [0.20, 0.40, 0.60, 0.80],
}


def score_absolute(value, cuts):
    """Map a raw value to 0-4 using upper-inclusive cut points."""
    for i, c in enumerate(cuts):
        if value <= c:
            return i
    return 4


def score_in_out_ratio(value):
    """
    in_out_ratio → 0-4 by ROUNDING the ratio to the nearest integer
    (project decision #2: 2.5→3, 2.4→2), clamped to [0,4].
    Uses round-half-UP (not banker's rounding) so 2.5 → 3 exactly.
    """
    return int(min(4, max(0, math.floor(float(value) + 0.5))))


def score_rank(series):
    """
    RANK-normalize a metric to a within-group percentile, then map to 0-4 with
    the shared percentile bands from README_metrics.pdf:
        0.00-0.20→0 · 0.21-0.40→1 · 0.41-0.60→2 · 0.61-0.80→3 · 0.81-1.00→4
    Used for MATTR, betweenness_centrality, response_consistency.
    Source for K-means/percentile validity of small-group ranking is downstream;
    the bands themselves are defined in README_metrics.pdf.
    """
    pct = series.rank(pct=True)
    cuts = [0.20, 0.40, 0.60, 0.80]

    def to_band(p):
        for i, c in enumerate(cuts):
            if p <= c:
                return i
        return 4
    return pct.apply(to_band)


def score_features(feats, mode="absolute"):
    """
    Convert the 13 raw metrics to integer 0-4 scores.

    mode="absolute" (default, used for cross-dataset / ML training):
        10 metrics → absolute thresholds (Slides 2 & 4),
        3 metrics  → within-group rank percentile (README).
        Gives values comparable across the 5 / 148 / 1,639-member groups.

    mode="rank_all" (small-team report mode — project decision, June 2026):
        ALL 13 metrics → within-group rank percentile → 0-4.
        Extends the README's small-group rationale (rank, don't threshold) from
        the 3 network/linguistic metrics to every metric, because on a tiny,
        homogeneous 5-person team the absolute bands collapse everyone into the
        same band (zero discriminative power). This restores contrast between
        members. NOTE: it sacrifices absolute cross-dataset comparability, so it
        is used ONLY for the team's own report, never for ML training.

    Returns a DataFrame with `author` + 13 integer (0-4) score columns.
    """
    assert mode in ("absolute", "rank_all"), f"unknown scoring mode {mode!r}"
    scores = pd.DataFrame({"author": feats["author"].values})

    if mode == "rank_all":
        for m in ALL_METRICS:
            scores[m] = score_rank(feats[m]).values
        return scores[["author"] + ALL_METRICS]

    # ── absolute mode ──
    scores["in_out_ratio"] = feats["in_out_ratio"].apply(score_in_out_ratio)
    for m in ABSOLUTE_METRICS:
        if m == "in_out_ratio":
            continue
        scores[m] = feats[m].apply(lambda v, c=ABSOLUTE_BANDS[m]: score_absolute(v, c))
    for m in RANK_METRICS:
        scores[m] = score_rank(feats[m]).values
    return scores[["author"] + ALL_METRICS]


# ════════════════════════════════════════════════════════════════════════════
# SLIDE-6 CLASSIFICATION TABLE  (expected 0-4 score range per archetype/metric)
# Statusmeeting3_team5.pptx, Slide 6. Ranges are inclusive [lo, hi].
# ════════════════════════════════════════════════════════════════════════════
SLIDE6 = {
    #  metric                : (Bee,        Ant,        Leech,      Butterfly,  Capybara)
    "in_out_ratio":           [(0.0, 0.8), (0.5, 1.5), (2.5, 4.0), (0.0, 1.5), (1.2, 2.9)],
    "avg_msg_length":         [(1.0, 2.0), (0.5, 1.5), (0.0, 0.5), (1.0, 4.0), (0.5, 1.5)],
    "replies_sent":           [(1.0, 4.0), (1.5, 3.5), (0.0, 0.5), (1.5, 2.5), (2.0, 3.0)],
    "question_ratio":         [(1.5, 2.5), (0.0, 0.5), (0.0, 0.5), (0.5, 1.5), (0.0, 4.0)],
    "avg_mentions":           [(2.0, 4.0), (0.0, 0.5), (0.0, 0.5), (1.0, 2.0), (0.5, 1.5)],
    "task_focus_score":       [(0.5, 2.5), (2.5, 4.0), (0.0, 0.5), (0.0, 0.5), (0.0, 1.0)],
    "harmony_score":          [(0.5, 1.5), (0.5, 1.5), (0.0, 0.5), (1.5, 3.5), (2.5, 4.0)],
    "emoji_ratio":            [(0.0, 0.5), (0.0, 0.5), (0.0, 0.5), (2.5, 4.0), (0.5, 3.5)],
    "emotion_density":        [(0.5, 1.5), (0.0, 1.0), (0.0, 1.0), (2.5, 4.0), (1.5, 3.0)],
    "initiation_rate":        [(2.0, 4.0), (0.5, 1.5), (0.0, 0.5), (2.5, 3.5), (0.0, 2.0)],
    "MATTR":                  [(3.0, 4.0), (0.5, 1.5), (0.0, 0.5), (1.0, 2.0), (0.5, 2.0)],
    "betweenness_centrality": [(3.5, 4.0), (0.5, 1.5), (0.0, 0.5), (1.0, 2.0), (0.5, 1.5)],
    "response_consistency":   [(0.5, 1.5), (3.5, 4.0), (0.0, 0.5), (1.5, 2.5), (2.0, 3.0)],
}
# Column order of the tuples above:
SLIDE6_ORDER = ["🐝 Bee", "🐜 Ant", "🔴 Leech", "🦋 Butterfly", "🦫 Capybara"]


_FRIENDLY = {
    "🐝 Bee": "bee", "🐜 Ant": "ant", "🔴 Leech": "leech",
    "🦋 Butterfly": "butterfly", "🦫 Capybara": "capybara",
}


def classify(scores):
    """
    Apply the Slide-6 logic to a 0-4 score table. For each person × archetype,
    count the metrics whose 0-4 score falls inside the archetype's expected range
    (`matches`, 0-13). Two complementary readouts are produced:

      • FIT score  <arch>_fit  = matches / 13 × 100   (project decision #1)
        An ABSOLUTE, interpretable 0-100 "how well does this member match this
        archetype" — NOT forced to sum to 100, so a strong match reads e.g. 62%
        instead of being compressed by the 5-way split.

      • DISTRIBUTION <arch>_pct = matches normalized across the 5 archetypes to
        sum to 100% — the relative profile (kept for the original Slide-6 view).

    Because both divide the same `matches` by a per-member constant, they give
    the SAME ranking, so the hard `archetype` label is identical either way.

    Returns: author + <arch>_fit (×5) + <arch>_pct (×5) + archetype + fit_top.
    """
    n_metrics = len(ALL_METRICS)
    rows = []
    for _, r in scores.iterrows():
        matches = {a: 0 for a in SLIDE6_ORDER}
        for m in ALL_METRICS:
            sc = r[m]
            for ai, arch in enumerate(SLIDE6_ORDER):
                lo, hi = SLIDE6[m][ai]
                if lo <= sc <= hi:
                    matches[arch] += 1
        fit   = {a: matches[a] / n_metrics * 100.0 for a in SLIDE6_ORDER}
        total = sum(matches.values())
        norm  = ({a: matches[a] / total * 100.0 for a in SLIDE6_ORDER}
                 if total > 0 else {a: 20.0 for a in SLIDE6_ORDER})
        row = {"author": r["author"]}
        for a in SLIDE6_ORDER:
            row[f"{_FRIENDLY[a]}_fit"] = fit[a]
            row[f"{_FRIENDLY[a]}_pct"] = norm[a]
        rows.append(row)

    out = pd.DataFrame(rows)
    fit_cols = [f"{_FRIENDLY[a]}_fit" for a in SLIDE6_ORDER]
    pct_cols = [f"{_FRIENDLY[a]}_pct" for a in SLIDE6_ORDER]
    # hard label from FIT (identical ordering to pct).
    fit_to_arch = {f"{_FRIENDLY[a]}_fit": a for a in SLIDE6_ORDER}
    out["archetype"] = out[fit_cols].idxmax(axis=1).map(fit_to_arch)
    out["fit_top"]   = out[fit_cols].max(axis=1)          # winner's fit % (0-100)
    keep = ["author"] + fit_cols + pct_cols + ["archetype", "fit_top"]
    return out[keep].round(1)


def sanity_check_scores(scores, label="dataset"):
    """Print the 0-4 score matrix and flag metrics with no variance across users."""
    print(f"\n── SANITY: 0-4 score matrix [{label}] " + "─" * 30)
    print(scores.set_index("author")[ALL_METRICS].to_string())
    # every score must be an integer in 0..4
    arr = scores[ALL_METRICS].to_numpy()
    assert ((arr >= 0) & (arr <= 4)).all(), f"[{label}] scores outside 0-4!"
    flat = [m for m in ALL_METRICS if scores[m].nunique() == 1]
    if flat:
        print(f"  ⚠ FLAT scores (same band for everyone, no discriminative power): {flat}")
    else:
        print("  ✓ all metrics discriminate between at least two users")


# ════════════════════════════════════════════════════════════════════════════
# Convenience: full rule pipeline on one clean.json
# ════════════════════════════════════════════════════════════════════════════
def rule_pipeline(json_path, label="dataset", mode="absolute", verbose=True):
    """Load → extract → sanity → score → sanity → classify. Returns (feats, scores, result)."""
    df = load_clean(json_path)
    feats = extract_features(df, label=label, verbose=verbose)
    if verbose:
        sanity_check_features(feats, label=label)
    scores = score_features(feats, mode=mode)
    if verbose:
        sanity_check_scores(scores, label=label)
    result = classify(scores)
    return feats, scores, result


if __name__ == "__main__":
    print("\n🧭  SocialCompass — Rule-based COIN classifier (Slide-6 logic)")
    print("=" * 64)
    # Small homogeneous team → use rank_all mode so every metric discriminates.
    feats, scores, result = rule_pipeline(
        "whatsapp_clean.json", label="Team WhatsApp", mode="rank_all")

    result.to_json("wa_rule_results.json", orient="records", lines=True)

    fit_cols = ["bee_fit", "ant_fit", "butterfly_fit", "capybara_fit", "leech_fit"]
    print("\n" + "=" * 64)
    print("  RULE-BASED archetype FIT (0-100, absolute match strength)")
    print("=" * 64)
    print(result[["author", "archetype", "fit_top"] + fit_cols].to_string(index=False))
    print("\n  Archetype counts:")
    print(result["archetype"].value_counts().to_string())
    print(f"\n  Winner-fit range: {result['fit_top'].min():.0f}% – {result['fit_top'].max():.0f}%  "
          f"(was ~24-29% under cross-archetype normalization)")
    print("  ✅ Saved → wa_rule_results.json")
