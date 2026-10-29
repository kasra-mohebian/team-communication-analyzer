"""
eval_vs_opus.py — how well does the model reproduce Opus's labels on held-out
Slack/Nankani users it never saw during training?

Method (honest test split, not in-sample):
  Stratified 5-fold over the 197 Opus-labeled users. Each user is predicted
  exactly ONCE, by a model trained only on the OTHER folds (out-of-fold / OOF).
  We aggregate all 197 OOF predictions and compare argmax to the Opus dominant
  label. This is the standard cross-validated test against the gold labels —
  every reported number is on data the model did not train on.

Reports for both the production FUSION model and the METRIC-ONLY baseline:
  • accuracy, macro-F1, weighted-F1
  • per-class precision / recall / F1 / support
  • confusion matrix (rows = Opus gold, cols = model)
  • soft-distribution agreement: KL, cosine
  • per-dataset accuracy (Slack vs Nankani test users)

Run:  python eval_vs_opus.py
"""
import json
import warnings
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

import train_ml as T
import optimize_model as O
import config

warnings.filterwarnings("ignore")
ARCH = T.ARCH_KEYS
EMO = {"bee": "🐝", "ant": "🐜", "butterfly": "🦋", "capybara": "🦫", "leech": "🔴"}


def oof_predictions(D, view, n_splits=5):
    """Out-of-fold soft predictions for every labeled user. view ∈ {fusion, metric, style}."""
    M = D["M_train"]; S = D["S_train"]; y = D["y"]; w = D["w"]; hard = D["hard"]
    oof = np.zeros_like(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=T.RANDOM_STATE)
    for tr, te in skf.split(M, hard):
        if view == "fusion":
            mdl = T.FusionModel(T.FUSION_METRIC_W).fit(M[tr], S[tr], y[tr], sample_weight=w[tr])
            oof[te] = mdl.predict(M[te], S[te])
        elif view == "metric":
            mdl = T.SoftRegressor("rf").fit(M[tr], y[tr], sample_weight=w[tr])
            oof[te] = mdl.predict(M[te])
        else:  # style
            mdl = T.SoftRegressor("rf").fit(S[tr], y[tr], sample_weight=w[tr])
            oof[te] = mdl.predict(S[te])
    return oof


def report(name, y_soft, oof, hard, datasets):
    pred = oof.argmax(1)
    acc = accuracy_score(hard, pred)
    maf1 = f1_score(hard, pred, average="macro", zero_division=0)
    wf1 = f1_score(hard, pred, average="weighted", zero_division=0)
    # soft agreement
    p, q = y_soft + T.EPS, oof + T.EPS
    kl = float(np.mean(np.sum(p * np.log(p / q), axis=1)))
    cos = float(np.mean(np.sum(y_soft * oof, 1) /
                        (np.linalg.norm(y_soft, axis=1) * np.linalg.norm(oof, axis=1) + T.EPS)))

    print("\n" + "═" * 70)
    print(f"  {name}")
    print("═" * 70)
    print(f"  Accuracy   : {acc*100:5.1f}%        (argmax vs Opus dominant label)")
    print(f"  Macro-F1   : {maf1:5.3f}        (every class weighted equally)")
    print(f"  Weighted-F1: {wf1:5.3f}        (weighted by class size)")
    print(f"  Soft KL    : {kl:5.3f}        cosine: {cos:5.3f}")

    print("\n  Per-class (precision / recall / F1 / support):")
    rep = classification_report(hard, pred, labels=list(range(5)),
                                target_names=[f"{EMO[a]} {a}" for a in ARCH],
                                output_dict=True, zero_division=0)
    for a in ARCH:
        key = f"{EMO[a]} {a}"
        r = rep[key]
        flag = "  ⚠ rare" if r["support"] < 15 else ""
        print(f"    {key:<13} P={r['precision']:.2f}  R={r['recall']:.2f}  "
              f"F1={r['f1-score']:.2f}  n={int(r['support'])}{flag}")

    print("\n  Confusion matrix (row = Opus gold ↓ , col = model →):")
    cm = confusion_matrix(hard, pred, labels=list(range(5)))
    print("            " + "".join(f"{EMO[a]:>6}" for a in ARCH))
    for i, a in enumerate(ARCH):
        print(f"    {EMO[a]} {a:<8}" + "".join(f"{cm[i][j]:>6}" for j in range(5)))

    # per-dataset accuracy
    print("\n  Accuracy by source dataset:")
    for ds in ["slack", "nankani"]:
        mask = np.array(datasets) == ds
        if mask.any():
            a = accuracy_score(hard[mask], pred[mask])
            print(f"    {ds:<9} n={int(mask.sum()):<4} acc={a*100:.1f}%")
    return {"accuracy": acc, "macro_f1": maf1, "weighted_f1": wf1, "kl": kl, "cosine": cos}


def main():
    print("Loading features (metric + text-style) for the 197 Opus-labeled users…")
    D = O.load_everything()
    hard = D["hard"]; y = D["y"]
    datasets = list(D["merged"]["dataset"])

    counts = np.bincount(hard, minlength=5)
    print("  Opus label distribution (the gold we test against):")
    for k, a in enumerate(ARCH):
        print(f"    {EMO[a]} {a:<10} {counts[k]:>4}  ({counts[k]/len(hard)*100:4.1f}%)")
    print("\n  Random guess accuracy ≈ {:.1f}%   "
          "Majority-class (bee) ≈ {:.1f}%".format(100/5, counts.max()/len(hard)*100))

    out = {}
    out["metric_only"] = report("METRIC-ONLY (13 network metrics)",
                                 y, oof_predictions(D, "metric"), hard, datasets)
    out["style_only"] = report("TEXT-STYLE-ONLY (12 signals)",
                                y, oof_predictions(D, "style"), hard, datasets)
    out["fusion"] = report("★ FUSION (production: 0.15·metric + 0.85·style)",
                           y, oof_predictions(D, "fusion"), hard, datasets)

    print("\n" + "═" * 70)
    print("  SUMMARY (out-of-fold, vs Opus labels)")
    print("═" * 70)
    print(f"  {'model':<28}{'acc':>8}{'macroF1':>10}{'wF1':>8}{'KL':>8}{'cos':>8}")
    for k, lbl in [("metric_only", "metric-only"),
                   ("style_only", "style-only"),
                   ("fusion", "★ fusion (production)")]:
        r = out[k]
        print(f"  {lbl:<28}{r['accuracy']*100:>7.1f}%{r['macro_f1']:>10.3f}"
              f"{r['weighted_f1']:>8.3f}{r['kl']:>8.3f}{r['cosine']:>8.3f}")

    json.dump(out, open(config.EVAL_VS_OPUS, "w"), indent=2)
    print("\n  → wrote eval_vs_opus.json")


if __name__ == "__main__":
    main()
