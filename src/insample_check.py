"""
insample_check.py — Sanity check: run the saved production model on its own training data.

Expected outcome: in-sample soft-acc >> held-out (WhatsApp) soft-acc.
If in-sample ≈ held-out the model is underfit; if in-sample ≫ held-out it overfit.
Results saved to outputs/results/insample_check.json.
"""

import sys, json
import numpy as np
import joblib

sys.path.insert(0, "src")
import config

# These must be in __main__ namespace so joblib can deserialize the saved model
# (model was pickled while train_ml.py was __main__, so classes are stored as __main__.*)
from train_ml import (
    load_training_matrix, load_style_features,
    soft_metrics, ARCH_KEYS, CONF_WEIGHT,
    FusionModel, SoftRegressor, MeanBaseline, LogRegBaseline,
)


def hard_metrics(y_true_hard, y_pred_hard, n_classes=5, arch_keys=ARCH_KEYS):
    acc = float(np.mean(y_true_hard == y_pred_hard))
    per_class = {}
    for k, name in enumerate(arch_keys):
        tp = int(np.sum((y_true_hard == k) & (y_pred_hard == k)))
        fn = int(np.sum((y_true_hard == k) & (y_pred_hard != k)))
        fp = int(np.sum((y_true_hard != k) & (y_pred_hard == k)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[name] = {"n": int(tp + fn), "precision": round(prec, 4),
                           "recall": round(rec, 4), "f1": round(f1, 4)}
    return acc, per_class


def main():
    print("\n🔍  In-sample sanity check")
    print("=" * 50)

    # Load saved model
    bundle = joblib.load(config.ML_MODEL)
    model  = bundle["model"]
    fusion_w = bundle["fusion_metric_w"]
    print(f"  Model loaded: fusion metric_w={fusion_w:.2f} / style_w={1-fusion_w:.2f}")

    # Load training data (same pipeline as train_ml.py)
    merged, X, y, w, hard_true, _ = load_training_matrix()
    S = load_style_features(merged)
    print(f"  Training users: {len(X)}")

    # Predict on ALL training data
    train_pred = model.predict(X, S)
    hard_pred  = train_pred.argmax(axis=1)

    # Soft metrics
    kl, cos, brier, sacc = soft_metrics(y, train_pred)

    # Hard metrics
    hard_acc, per_class = hard_metrics(hard_true, hard_pred)

    # Per-class soft-accuracy
    per_class_sacc = {}
    for k, name in enumerate(ARCH_KEYS):
        mask = hard_true == k
        if mask.sum() > 0:
            _, _, _, s = soft_metrics(y[mask], train_pred[mask])
            per_class_sacc[name] = round(s, 4)

    print(f"\n  ── In-sample (training data) ──────────────────")
    print(f"  Soft-accuracy : {sacc:.4f}  ({sacc*100:.1f}%)")
    print(f"  Cosine sim    : {cos:.4f}")
    print(f"  KL divergence : {kl:.4f}")
    print(f"  Hard accuracy : {hard_acc:.4f}  ({hard_acc*100:.1f}%)")
    print(f"\n  ── Held-out WhatsApp (reference) ──────────────")

    # Load held-out results if available
    holdout = {}
    try:
        with open(config.RESULTS / "holdout_comparison.json") as f:
            holdout = json.load(f)
        ho_sacc = holdout.get("soft_accuracy") or holdout.get("sacc")
        ho_cos  = holdout.get("cosine") or holdout.get("cos")
        if ho_sacc:
            print(f"  Soft-accuracy : {ho_sacc:.4f}  ({ho_sacc*100:.1f}%)")
        if ho_cos:
            print(f"  Cosine sim    : {ho_cos:.4f}")
    except Exception:
        print("  (holdout_comparison.json not found — skipping reference)")

    print(f"\n  ── Per-class in-sample soft-accuracy ──────────")
    for name, s in per_class_sacc.items():
        bar = "█" * int(s * 30)
        print(f"  {name:<12} {s:.3f}  {bar}")

    # Save results
    out = {
        "n_train": int(len(X)),
        "fusion_metric_w": float(fusion_w),
        "insample": {
            "soft_accuracy": round(sacc, 4),
            "cosine": round(cos, 4),
            "kl_divergence": round(kl, 4),
            "brier_score": round(brier, 4),
            "hard_accuracy": round(hard_acc, 4),
            "per_class_f1": per_class,
            "per_class_soft_accuracy": per_class_sacc,
        },
        "holdout_reference": holdout if holdout else "not_found",
        "interpretation": {
            "insample_vs_holdout": (
                f"in-sample soft-acc {sacc:.3f} vs held-out {holdout.get('soft_accuracy', '?')} "
                f"— gap = {sacc - holdout.get('soft_accuracy', sacc):.3f}"
                if holdout.get("soft_accuracy") else "holdout data not available"
            )
        }
    }

    import pathlib
    out_path = pathlib.Path(config.RESULTS) / "insample_check.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ✅  Saved → {out_path}")


if __name__ == "__main__":
    main()
