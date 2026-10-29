"""
ab_features.py — A/B test: v1 (broken interaction metrics) vs v2 (fixed).

Text view (style + embeddings) is identical in both arms, so any difference is
attributable solely to the 5 corrected interaction metrics. Reports, per arm:
  • metric-only OOF: accuracy, macro-F1, per-class F1   (does the metric view
    predict Opus labels better once the interaction metrics are fixed?)
  • fusion OOF: same                                     (production model)
  • team blind held-out: agreement + per-member
  • fusion-weight sweep on team agreement + CV-KL        (can we now trust the
    metric view enough to raise its 0.15 weight?)
"""
import json, warnings, numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import train_ml as T
import optimize_model as O
import eval_vs_opus as E
import config

warnings.filterwarnings("ignore")
ARCH = T.ARCH_KEYS
RS = 42


def per_class_f1(hard, pred):
    return f1_score(hard, pred, labels=list(range(5)), average=None, zero_division=0)


def eval_arm(train_file, team_file):
    T.TRAIN_FEATURES_FILE = train_file
    T.TEAM_FEATURES_FILE = team_file
    D = O.load_everything()
    hard = D["hard"]
    res = {}
    for view in ("metric", "fusion"):
        oof = E.oof_predictions(D, view)
        pred = oof.argmax(1)
        res[view] = {
            "acc": accuracy_score(hard, pred),
            "macro_f1": f1_score(hard, pred, average="macro", zero_division=0),
            "per_class": dict(zip(ARCH, per_class_f1(hard, pred))),
        }
    # team held-out (fusion at production weight)
    Xtr, Xte = O.assemble(D, {"M"}, hygiene=False)
    Str, Ste = D["S_train"], D["S_team"]
    mm = T.SoftRegressor("rf").fit(Xtr, D["y"], sample_weight=D["w"])
    ms = T.SoftRegressor("rf").fit(Str, D["y"], sample_weight=D["w"])
    pm, ps = mm.predict(Xte), ms.predict(Ste)
    # sweep fusion weight on team agreement + CV-KL
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RS)
    sweep = {}
    for a in (0.0, 0.15, 0.3, 0.5):
        blend = T.to_simplex(a * pm + (1 - a) * ps)
        hits = sum(ARCH[blend[i].argmax()] == D["gold"][D["team_authors"][i]]["archetype"]
                   for i in range(5))
        kls = []
        for tr, te in cv.split(Xtr, hard):
            am = T.SoftRegressor("rf").fit(Xtr[tr], D["y"][tr], sample_weight=D["w"][tr]).predict(Xtr[te])
            as_ = T.SoftRegressor("rf").fit(Str[tr], D["y"][tr], sample_weight=D["w"][tr]).predict(Str[te])
            kl, _, _ = T.soft_metrics(D["y"][te], T.to_simplex(a * am + (1 - a) * as_))
            kls.append(kl)
        sweep[a] = {"team_agree": hits / 5, "cv_kl": float(np.mean(kls))}
    blend15 = T.to_simplex(0.15 * pm + (1 - 0.15) * ps)
    res["team"] = {
        "members": [(D["team_authors"][i], D["gold"][D["team_authors"][i]]["archetype"],
                     ARCH[blend15[i].argmax()]) for i in range(5)],
        "sweep": sweep,
    }
    return res


def main():
    print("Evaluating v1 (broken) …")
    v1 = eval_arm(config.TRAIN_FEATURES, config.WA_FEATURES)
    print("Evaluating v2 (fixed) …")
    v2 = eval_arm(config.TRAIN_FEATURES_V2, config.WA_FEATURES_V2)

    def line(name, a, b, pct=False):
        s = "%+.1f" % ((b - a) * (100 if pct else 1))
        f = "{:.1f}%" if pct else "{:.3f}"
        return f"  {name:<22}{f.format(a*(100 if pct else 1)):>9}{f.format(b*(100 if pct else 1)):>9}{s:>9}"

    print("\n" + "=" * 56)
    print(f"  A/B: interaction-metric fix   {'v1':>9}{'v2':>9}{'Δ':>9}")
    print("=" * 56)
    for view in ("metric", "fusion"):
        print(f"\n  [{view.upper()} view]")
        print(line("accuracy", v1[view]["acc"], v2[view]["acc"], pct=True))
        print(line("macro-F1", v1[view]["macro_f1"], v2[view]["macro_f1"]))
        for k in ARCH:
            print(line(f"F1 {k}", v1[view]["per_class"][k], v2[view]["per_class"][k]))

    print("\n  [TEAM blind held-out — fusion @0.15]")
    for (n, g, p1), (_, _, p2) in zip(v1["team"]["members"], v2["team"]["members"]):
        f1 = "OK" if g == p1 else "XX"; f2 = "OK" if g == p2 else "XX"
        print(f"    {n:<8} gold={g:<9} v1:{f1} {p1:<9} v2:{f2} {p2}")
    a1 = sum(g == p for _, g, p in v1["team"]["members"]) / 5
    a2 = sum(g == p for _, g, p in v2["team"]["members"]) / 5
    print(f"    agreement: v1 {a1*100:.0f}%  ->  v2 {a2*100:.0f}%")

    print("\n  [Fusion-weight sweep: can we trust the metric view more now?]")
    print(f"    {'metric_w':<10}{'v1 agree/CV-KL':<20}{'v2 agree/CV-KL'}")
    for a in (0.0, 0.15, 0.3, 0.5):
        s1, s2 = v1["team"]["sweep"][a], v2["team"]["sweep"][a]
        print(f"    {a:<10}{s1['team_agree']*100:>3.0f}% / {s1['cv_kl']:.3f}      "
              f"{s2['team_agree']*100:>3.0f}% / {s2['cv_kl']:.3f}")

    json.dump({"v1": v1, "v2": v2}, open(config.AB_RESULTS, "w"),
              indent=2, default=float)
    print("\n  → ab_features_results.json")


if __name__ == "__main__":
    main()
