"""
train_and_predict.py
════════════════════
ML layer of the hybrid COIN archetype classifier (SocialCompass).

Architecture (agreed decisions, June 2026):
  TRAIN  : slack_clean.json (Chatterjee 2020) + nankani_clean.json (CODERS 2020)
  TEST   : whatsapp_clean.json (the team's own 5-member chat)

  STEP 1  Extract the 13 metrics (features.py) for all three datasets.
  STEP 2  Rank-normalize features WITHIN each dataset → comparable across the
          5 / 148 / 1,639-member groups (in_out_ratio uses received/sent, no N-1,
          for exactly this cross-dataset comparability).
  STEP 3  Rule-based pseudo-labels: score 0-4 + apply the Slide-6 ranges
          (classify_archetypes.py). These labels are the Random Forest's
          training target — they are interpretable and scientifically grounded
          (sna.pages decision #4), unlike raw unsupervised clusters.
  STEP 4  K-MEANS (K=5) as an INDEPENDENT unsupervised validation:
          report silhouette score (Rousseeuw 1987) and the Adjusted Rand Index
          between clusters and rule labels (do the data fall into 5 groups, and
          do those groups agree with the rules?).
  STEP 5  RANDOM FOREST (Breiman 2001) trained on the rule pseudo-labels;
          report 5-fold cross-validation accuracy + macro-F1 + feature importance.
  STEP 6  DOMAIN-SHIFT check between train and test feature distributions.
  STEP 7  Apply to the team: ML prediction + rule label + 50/50 HYBRID vote with
          a confidence score (high where rules & ML agree, flagged where not).

Scientific sources cited inline. Required sanity checks: feature distributions,
range asserts, near-zero variance flags, silhouette, CV accuracy, domain-shift
warning.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans                              # Lloyd (1957); Rousseeuw (1987) silhouette
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier            # Breiman, L. (2001). Random Forests. ML 45(1):5-32.
from sklearn.model_selection import cross_val_score, StratifiedKFold

from features import (
    ALL_METRICS, load_clean, extract_features, sanity_check_features,
)
from classify_archetypes import (
    ARCHETYPES, score_features, classify, sanity_check_scores, SLIDE6_ORDER,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
PCT_COLS = ["bee_pct", "ant_pct", "butterfly_pct", "capybara_pct", "leech_pct"]
PCT_TO_ARCH = {  # map friendly pct column → emoji archetype
    "bee_pct": "🐝 Bee", "ant_pct": "🐜 Ant", "butterfly_pct": "🦋 Butterfly",
    "capybara_pct": "🦫 Capybara", "leech_pct": "🔴 Leech",
}
PCT_TO_ARCH_INV = {  # emoji archetype → friendly key
    "🐝 Bee": "bee", "🐜 Ant": "ant", "🦋 Butterfly": "butterfly",
    "🦫 Capybara": "capybara", "🔴 Leech": "leech",
}


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def rank_normalize(feats):
    """Rank-normalize each metric to a within-dataset percentile (0-1)."""
    out = feats.copy()
    for m in ALL_METRICS:
        out[m] = feats[m].rank(pct=True)
    return out


def load_extract_score(path, label, mode="absolute"):
    """
    clean.json → raw feats → sanity → rank-norm → 0-4 scores → rule labels.
    mode="absolute" for the training datasets (cross-dataset comparability);
    mode="rank_all" for the small team report (better member separation).
    """
    print(f"\n{'═'*64}\n  DATASET: {label}  (scoring mode: {mode})\n{'═'*64}")
    df = load_clean(path)
    raw = extract_features(df, label=label)
    sanity_check_features(raw, label=label)
    scores = score_features(raw, mode=mode)
    rule = classify(scores)                       # fit + pct distribution + label
    norm = rank_normalize(raw)                    # RF / K-means input space
    return raw, norm, scores, rule


def domain_shift_report(train_raw, test_raw, label_train, label_test, d_thresh=0.8):
    """
    Flag features whose distribution shifts between train and test using a
    standardized mean difference (Cohen's d). |d| > 0.8 = large shift
    (Cohen 1988). A large shift means the RF is extrapolating, not interpolating.
    """
    print(f"\n── SANITY: domain shift  {label_train} → {label_test} " + "─" * 18)
    shifted = []
    for m in ALL_METRICS:
        a, b = train_raw[m].values, test_raw[m].values
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(b) > 1 else a.std() + 1e-9
        d = (b.mean() - a.mean()) / (pooled + 1e-12)
        marker = "  ⚠ SHIFT" if abs(d) > d_thresh else ""
        if abs(d) > d_thresh:
            shifted.append(m)
        print(f"    {m:<24} d = {d:+.2f}{marker}")
    if shifted:
        print(f"  ⚠ DOMAIN SHIFT on {len(shifted)}/{len(ALL_METRICS)} metrics: {shifted}")
        print("    → team test set differs from training groups on these signals; "
              "treat ML predictions with corresponding caution.")
    else:
        print("  ✓ no large domain shift detected (all |d| ≤ 0.8)")
    return shifted


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("\n🤖  SocialCompass — Hybrid COIN classifier (rule + ML)")
    print("=" * 64)

    # ── STEP 1-3 : extract / score / rule-label every dataset ───────────────
    slack_raw,  slack_norm,  _, slack_rule   = load_extract_score("slack_clean.json",   "Slack (Chatterjee 2020)",   mode="absolute")
    nank_raw,   nank_norm,   _, nank_rule     = load_extract_score("nankani_clean.json", "CODERS WhatsApp (Nankani 2020)", mode="absolute")
    # team uses small-group rank_all mode so the 5 members separate cleanly.
    wa_raw,     wa_norm,    wa_scores, wa_rule = load_extract_score("whatsapp_clean.json", "Team WhatsApp", mode="rank_all")
    sanity_check_scores(wa_scores, label="Team WhatsApp")

    # ── Build the combined training matrix ──────────────────────────────────
    slack_norm["archetype"] = slack_rule["archetype"].values
    nank_norm["archetype"]  = nank_rule["archetype"].values
    slack_norm["dataset"], nank_norm["dataset"] = "slack", "nankani"

    train = pd.concat(
        [slack_norm[ALL_METRICS + ["archetype", "dataset"]],
         nank_norm[ALL_METRICS + ["archetype", "dataset"]]],
        ignore_index=True,
    )
    train_raw = pd.concat([slack_raw[ALL_METRICS], nank_raw[ALL_METRICS]], ignore_index=True)
    print(f"\n  Combined training set: {len(train):,} users "
          f"({len(slack_norm):,} Slack + {len(nank_norm):,} Nankani)")
    print("  Rule-label distribution (RF training target):")
    for arch, cnt in train["archetype"].value_counts().items():
        print(f"    {arch:<16} {cnt:>5}  ({cnt/len(train)*100:4.1f}%)")

    X = train[ALL_METRICS].to_numpy()

    # ── STEP 4 : K-MEANS unsupervised validation ────────────────────────────
    print(f"\n── ML VALIDATION: K-means (K=5) " + "─" * 33)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=5, n_init=10, random_state=RANDOM_STATE)
    clusters = km.fit_predict(Xs)
    sil = silhouette_score(Xs, clusters)          # Rousseeuw (1987)
    ari = adjusted_rand_score(
        LabelEncoder().fit_transform(train["archetype"]), clusters
    )
    print(f"  Silhouette score (Rousseeuw 1987): {sil:.3f}   "
          f"(>0.25 = some structure, >0.5 = strong)")
    print(f"  Adjusted Rand Index, clusters vs rule labels: {ari:.3f}   "
          f"(0 = chance, 1 = identical)")
    if sil < 0.25:
        print("  ⚠ low silhouette: the 13-D feature space does not split cleanly "
              "into 5 spheres — expected, archetypes overlap; rules resolve the overlap.")

    # ── STEP 5 : RANDOM FOREST on rule pseudo-labels ────────────────────────
    print(f"\n── ML MODEL: Random Forest (Breiman 2001) " + "─" * 22)
    le = LabelEncoder()
    y  = le.fit_transform(train["archetype"])
    rf_params = dict(n_estimators=300, max_depth=12, min_samples_leaf=3,
                     class_weight="balanced", random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(RandomForestClassifier(**rf_params), X, y, cv=cv, scoring="accuracy")
    cv_f1  = cross_val_score(RandomForestClassifier(**rf_params), X, y, cv=cv, scoring="f1_macro")
    print(f"  5-fold CV accuracy : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  5-fold CV macro-F1 : {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    rf = RandomForestClassifier(**rf_params).fit(X, y)
    print(f"  Train accuracy     : {rf.score(X, y)*100:.1f}%")
    print("  Feature importance (top 8):")
    for feat, imp in sorted(zip(ALL_METRICS, rf.feature_importances_),
                            key=lambda t: t[1], reverse=True)[:8]:
        print(f"    {feat:<24} {imp:.3f}  {'█'*int(imp*60)}")

    # ── STEP 6 : DOMAIN-SHIFT check ─────────────────────────────────────────
    domain_shift_report(train_raw, wa_raw[ALL_METRICS], "Train(Slack+Nankani)", "Team WA")

    # ── STEP 7 : predict for the team + hybrid vote ─────────────────────────
    print(f"\n── PREDICT + HYBRID VOTE (50% rule / 50% ML) " + "─" * 20)
    X_wa = wa_norm[ALL_METRICS].to_numpy()
    ml_proba = rf.predict_proba(X_wa)
    ml_label = le.inverse_transform(rf.predict(X_wa))

    # rule distribution per member (already normalized to 100% by classify()).
    rule_dist = wa_rule.set_index("author")[PCT_COLS]

    arch_key = {a: PCT_TO_ARCH_INV[a] for a in ARCHETYPES}   # emoji → bee/ant/...
    final_arch, final_conf, agree_flags = [], [], []
    ml_rows, hyb_rows = [], []                               # full decision trace
    for i, author in enumerate(wa_rule["author"]):
        ml_d   = {le.classes_[j]: ml_proba[i][j] for j in range(len(le.classes_))}
        rule_d = {PCT_TO_ARCH[c]: rule_dist.loc[author, c] / 100.0 for c in PCT_COLS}
        hybrid = {a: 0.5 * ml_d.get(a, 0.0) + 0.5 * rule_d.get(a, 0.0) for a in ARCHETYPES}
        winner = max(hybrid, key=hybrid.get)
        final_arch.append(winner)
        final_conf.append(round(hybrid[winner] * 100, 1))
        agree_flags.append(wa_rule.loc[wa_rule["author"] == author, "archetype"].iloc[0] == ml_label[i])
        # per-archetype ML probability and hybrid score (0-100) for the debug tab
        ml_rows.append({f"ml_{arch_key[a]}": round(ml_d.get(a, 0.0) * 100, 1) for a in ARCHETYPES})
        hyb_rows.append({f"hyb_{arch_key[a]}": round(hybrid[a] * 100, 1) for a in ARCHETYPES})

    # ── Assemble final results (dashboard-compatible) ───────────────────────
    res = wa_rule.copy()
    res["archetype_rule"] = wa_rule["archetype"]
    res["archetype_ml"]   = ml_label
    res["ml_confidence"]  = [round(p.max() * 100, 1) for p in ml_proba]
    res["archetype"]      = final_arch            # final hybrid label
    res["confidence"]     = final_conf
    res["rule_ml_agree"]  = agree_flags
    # full decision trace: per-archetype ML probability + hybrid score (for debug tab)
    for c in ml_rows[0]:
        res[c] = [row[c] for row in ml_rows]
    for c in hyb_rows[0]:
        res[c] = [row[c] for row in hyb_rows]
    # attach raw + 0-4 score columns the dashboard / paper may want
    for m in ALL_METRICS:
        res[m] = wa_raw.set_index("author").loc[res["author"], m].values
    # legacy score-column aliases used by dashboard.py
    res["bee_score"]         = res["bee_pct"]
    res["ant_score"]         = res["ant_pct"]
    res["butterfly_score_n"] = res["butterfly_pct"]
    res["capybara_score_n"]  = res["capybara_pct"]
    res["leech_risk"]        = res["leech_pct"]

    res.to_json("wa_results.json", orient="records", lines=True)

    # training reference (raw features + rule label) for the paper.
    train_save = pd.concat([slack_raw.assign(dataset="slack"),
                            nank_raw.assign(dataset="nankani")], ignore_index=True)
    train_save["archetype"] = train["archetype"].values
    train_save.to_json("training_results.json", orient="records", lines=True)

    # ── Final summary table ─────────────────────────────────────────────────
    agree_n = sum(agree_flags)
    print("\n" + "=" * 64)
    print("  FINAL — hybrid archetype distribution per member")
    print("=" * 64)
    print(f"  {'Name':<10}{'Hybrid label':<16}{'🐝':>6}{'🐜':>6}{'🦋':>6}{'🦫':>6}{'🔴':>6}  Fit  Conf R=ML")
    for _, r in res.iterrows():
        mark = "✓" if r["rule_ml_agree"] else "✗"
        print(f"  {str(r['author'])[:9]:<10}{r['archetype']:<16}"
              f"{r['bee_pct']:>5.0f}%{r['ant_pct']:>5.0f}%{r['butterfly_pct']:>5.0f}%"
              f"{r['capybara_pct']:>5.0f}%{r['leech_pct']:>5.0f}% {r['fit_top']:>4.0f}% {r['confidence']:>4.0f}%  {mark}")

    print(f"\n  Convergent validity: rule == ML for {agree_n}/{len(res)} members")
    print(f"  Train: {len(train):,} users · CV accuracy {cv_acc.mean():.3f} · "
          f"macro-F1 {cv_f1.mean():.3f} · silhouette {sil:.3f}")
    print("  ✅ Saved → wa_results.json,  training_results.json")
    print("\n  Run:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
