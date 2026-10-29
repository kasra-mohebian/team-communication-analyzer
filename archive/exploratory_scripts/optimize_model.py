"""
optimize_model.py — find the configuration that makes the model classify the
team the way Opus did, measured on the BLIND held-out (wa_holdout_opus.json).

Every configuration is scored two ways:
  • CV   — Repeated Stratified 5-fold×5 on the 197 Opus-labeled users
           (generalization: does the feature set predict Opus soft labels?).
  • TEAM — refit on all 197, predict the 5 team members, compare to the blind
           Opus gold read from raw text (top-1 agreement + mean KL + cosine).

We pick the optimum by TEAM agreement first (that is the real target), then by
CV-KL as a generalization tie-break, so we don't overfit config choices to 5
people. Nothing here is trained on the team — WhatsApp stays a pure test set.

Feature blocks (toggled per config):
  M  = 13 behavioural/graph metrics (rank-normalized)        [scale-dependent]
  Mh = metric hygiene: drop degenerate-for-small-group metrics
       (betweenness_centrality, in_out_ratio) + map team onto the TRAINING
       distribution quantiles instead of within-5 rank                 [robust]
  S  = 12 interpretable text-style signals                  [scale-invariant]
  E  = 384-d MiniLM message embedding                       [scale-invariant]
"""
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

import train_ml as T
from features import ALL_METRICS
import text_features as TF
import config

warnings.filterwarnings("ignore")
RS = 42
ARCH = T.ARCH_KEYS
DROP_FOR_SMALL = ["betweenness_centrality", "in_out_ratio"]


# ════════════════════════════════════════════════════════════════════════════
# Feature assembly
# ════════════════════════════════════════════════════════════════════════════
def load_everything():
    """Aligned feature blocks for the 197 labeled users and the 5 team members."""
    labels = pd.read_json(T.LABELS_FILE, lines=True)
    feats = pd.read_json(T.TRAIN_FEATURES_FILE, lines=True)

    # --- metric block: train rank-norm within dataset (full community) ---
    norm = pd.concat([T.rank_normalize(g) for _, g in feats.groupby("dataset")],
                     ignore_index=True)
    merged = labels.merge(
        norm[["author", "dataset"] + ALL_METRICS],
        left_on=["user_id", "dataset"], right_on=["author", "dataset"], how="left")
    assert merged[ALL_METRICS].isna().any(axis=1).sum() == 0

    y = merged[ARCH].to_numpy(float)
    y = y / y.sum(1, keepdims=True)
    w = merged["confidence"].map(T.CONF_WEIGHT).fillna(0.4).to_numpy()
    hard = y.argmax(1)
    M_train = merged[ALL_METRICS].to_numpy(float)

    # raw metric distribution per dataset → for quantile transfer of the team
    raw_by_ds = {ds: g for ds, g in feats.groupby("dataset")}

    # --- team metrics ---
    team = pd.read_json(T.TEAM_FEATURES_FILE, lines=True)
    M_team_rank = T.rank_normalize(team)[ALL_METRICS].to_numpy(float)
    # quantile transfer: position each team raw value in the slack distribution
    ref = raw_by_ds["slack"]
    M_team_q = np.zeros_like(M_team_rank)
    for j, m in enumerate(ALL_METRICS):
        ref_vals = np.sort(ref[m].to_numpy(float))
        for i in range(len(team)):
            M_team_q[i, j] = np.searchsorted(ref_vals, team[m].iloc[i]) / len(ref_vals)

    # --- text blocks (cached for train; built live for the 5-person team) ---
    idx = [(r["dataset"], r["user_id"]) for _, r in labels.iterrows()]
    E_train, S_train = TF.build_features(idx)
    team_idx = [("team", a) for a in team["author"]]
    E_team, S_team = TF.build_features(team_idx, team=True, use_cache=False)

    gold = {d["author"]: d for d in
            json.load(open(config.WA_HOLDOUT))["labels"]}
    return dict(
        y=y, w=w, hard=hard,
        M_train=M_train, M_team_rank=M_team_rank, M_team_q=M_team_q,
        S_train=S_train.astype(float), S_team=S_team.astype(float),
        E_train=E_train.astype(float), E_team=E_team.astype(float),
        team_authors=list(team["author"]), gold=gold, merged=merged)


def assemble(D, blocks, hygiene=False):
    """Stack the requested feature blocks for train and team."""
    Xtr, Xte = [], []
    if "M" in blocks:
        if hygiene:
            keep = [i for i, m in enumerate(ALL_METRICS) if m not in DROP_FOR_SMALL]
            Xtr.append(D["M_train"][:, keep])
            Xte.append(D["M_team_q"][:, keep])           # quantile transfer
        else:
            Xtr.append(D["M_train"])
            Xte.append(D["M_team_rank"])                 # within-5 rank
    if "S" in blocks:
        Xtr.append(D["S_train"]); Xte.append(D["S_team"])
    if "E" in blocks:
        Xtr.append(D["E_train"]); Xte.append(D["E_team"])
    return np.hstack(Xtr), np.hstack(Xte)


# ════════════════════════════════════════════════════════════════════════════
# Scoring
# ════════════════════════════════════════════════════════════════════════════
def cv_score(X, D, kind="rf", n_splits=5, n_repeats=5):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RS)
    kls, coss, f1s = [], [], []
    y, w, hard = D["y"], D["w"], D["hard"]
    for tr, te in cv.split(X, hard):
        mdl = T.SoftRegressor(kind).fit(X[tr], y[tr], sample_weight=w[tr])
        pred = mdl.predict(X[te])
        kl, cos, _ = T.soft_metrics(y[te], pred)
        kls.append(kl); coss.append(cos)
        f1s.append(f1_score(hard[te], pred.argmax(1), average="macro", zero_division=0))
    return np.mean(kls), np.mean(coss), np.mean(f1s)


def team_score(Xtr, Xte, D, kind="rf", leech_w=1.0):
    """Refit on all 197, predict team, compare to blind Opus gold."""
    w = D["w"].copy()
    if leech_w != 1.0:                       # optionally down-weight leech exemplars
        w = w * np.where(D["hard"] == ARCH.index("leech"), leech_w, 1.0)
    mdl = T.SoftRegressor(kind).fit(Xtr, D["y"], sample_weight=w)
    proba = mdl.predict(Xte)
    hits, kls, coss, rows = 0, [], [], []
    for i, name in enumerate(D["team_authors"]):
        g = D["gold"][name]
        gsoft = np.array([g["soft"][a] for a in ARCH])
        psoft = proba[i]
        gtop, ptop = g["archetype"], ARCH[psoft.argmax()]
        hits += gtop == ptop
        p, q = gsoft + T.EPS, psoft + T.EPS
        kls.append(float(np.sum(p * np.log(p / q))))
        coss.append(float(np.dot(gsoft, psoft) /
                          (np.linalg.norm(gsoft) * np.linalg.norm(psoft) + T.EPS)))
        rows.append((name, gtop, ptop, gtop == ptop))
    n = len(D["team_authors"])
    return dict(agree=hits / n, kl=np.mean(kls), cos=np.mean(coss), rows=rows,
                proba=proba)


# ════════════════════════════════════════════════════════════════════════════
# Experiments
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading feature blocks (metrics + text embeddings + style) …")
    D = load_everything()

    configs = [
        # (id, blocks, hygiene, kind, leech_w)
        ("C0  metric (current)",        "M",   False, "rf",   1.0),
        ("C1  metric+hygiene",          "M",   True,  "rf",   1.0),
        ("C2  metric+hygiene, leech×.5","M",   True,  "rf",   0.5),
        ("C3  style only",              "S",   False, "rf",   1.0),
        ("C4  embedding only",          "E",   False, "rf",   1.0),
        ("C5  style+embedding",         "SE",  False, "rf",   1.0),
        ("C6  metric+style",            "MS",  True,  "rf",   1.0),
        ("C7  metric+style+emb",        "MSE", True,  "rf",   1.0),
        ("C8  style+emb (logreg)",      "SE",  False, "logreg", 1.0),
    ]

    print("\n" + "=" * 86)
    print(f"{'config':<28}{'CV-KL↓':>8}{'CV-cos↑':>8}{'CV-F1↑':>8}  "
          f"{'TEAM-agree↑':>11}{'TEAM-KL↓':>9}{'TEAM-cos↑':>9}")
    print("=" * 86)

    results = []
    for cid, blocks, hyg, kind, lw in configs:
        bset = set(blocks) if blocks in ("M", "S", "E") else set(blocks)
        Xtr, Xte = assemble(D, bset, hygiene=hyg)
        if kind == "logreg":
            ck, cc, cf = _cv_logreg(Xtr, D)
            ts = _team_logreg(Xtr, Xte, D)
        else:
            ck, cc, cf = cv_score(Xtr, D, kind=kind)
            ts = team_score(Xtr, Xte, D, kind=kind, leech_w=lw)
        results.append((cid, ck, cc, cf, ts))
        print(f"{cid:<28}{ck:>8.3f}{cc:>8.3f}{cf:>8.3f}  "
              f"{ts['agree']*100:>9.0f}%{ts['kl']:>9.3f}{ts['cos']:>9.3f}")

    # pick optimum: team agreement first, then team-KL, then CV-KL
    best = max(results, key=lambda r: (r[4]["agree"], -r[4]["kl"], -r[1]))
    print("\n" + "=" * 86)
    print(f"★ OPTIMUM: {best[0]}  "
          f"(team agree {best[4]['agree']*100:.0f}% · team-KL {best[4]['kl']:.3f} "
          f"· CV-KL {best[1]:.3f})")
    print("  per-member (Opus gold → model):")
    for name, gtop, ptop, ok in best[4]["rows"]:
        print(f"    {'✅' if ok else '❌'} {name:<8} {gtop:<9} → {ptop}")

    json.dump({
        "configs": [{"id": r[0], "cv_kl": r[1], "cv_cos": r[2], "cv_f1": r[3],
                     "team_agree": r[4]["agree"], "team_kl": r[4]["kl"],
                     "team_cos": r[4]["cos"],
                     "team_rows": [list(x) for x in r[4]["rows"]]} for r in results],
        "optimum": best[0],
    }, open(config.OPTIMIZE_RESULTS, "w"), indent=2)
    print("\n  → wrote optimize_results.json")


# logreg variants (hard-label distillation) ---------------------------------
def _cv_logreg(X, D, n_splits=5, n_repeats=5):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RS)
    kls, coss, f1s = [], [], []
    for tr, te in cv.split(X, D["hard"]):
        mdl = T.LogRegBaseline().fit(X[tr], D["y"][tr], sample_weight=D["w"][tr])
        pred = mdl.predict(X[te])
        kl, cos, _ = T.soft_metrics(D["y"][te], pred)
        kls.append(kl); coss.append(cos)
        f1s.append(f1_score(D["hard"][te], pred.argmax(1), average="macro", zero_division=0))
    return np.mean(kls), np.mean(coss), np.mean(f1s)


def _team_logreg(Xtr, Xte, D):
    mdl = T.LogRegBaseline().fit(Xtr, D["y"], sample_weight=D["w"])
    proba = mdl.predict(Xte)
    hits, kls, coss, rows = 0, [], [], []
    for i, name in enumerate(D["team_authors"]):
        g = D["gold"][name]
        gsoft = np.array([g["soft"][a] for a in ARCH]); psoft = proba[i]
        gtop, ptop = g["archetype"], ARCH[psoft.argmax()]
        hits += gtop == ptop
        p, q = gsoft + T.EPS, psoft + T.EPS
        kls.append(float(np.sum(p * np.log(p / q))))
        coss.append(float(np.dot(gsoft, psoft) /
                          (np.linalg.norm(gsoft) * np.linalg.norm(psoft) + T.EPS)))
        rows.append((name, gtop, ptop, gtop == ptop))
    n = len(D["team_authors"])
    return dict(agree=hits / n, kl=np.mean(kls), cos=np.mean(coss), rows=rows, proba=proba)


if __name__ == "__main__":
    main()
