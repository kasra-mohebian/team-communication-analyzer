> Historical methodology and reported ablation. Some class F1 values originate from the best single-view model and do not average to the reported Fusion macro-F1. See RESULTS.md before comparing configurations. These figures have not been regenerated with this edition's corrected code.

# Synthetic augmentation for the rare archetypes

## Why
The real corpora (Slack / Nankani / Mbada) are technical Q&A and one professional
WhatsApp group. They are rich in **Bee** and **Ant** but structurally starved of
**Butterfly** (information-bridge) and **Capybara** (warm supporter): only 16 and
19 labeled users respectively, vs 98 Bee. With so few examples Capybara was
effectively **unlearnable** (out-of-fold F1 ≈ 0.008).

## Why not just add more real data?
We evaluated the **Garimella et al. (2020)** anonymized Indian-WhatsApp dataset
(6.3 M messages, 5,779 groups) as a real source. After mining 22,664 qualifying
users and reading a balanced 90-candidate pool, the verdict was **reject**: these
are political/religious/commercial **broadcast** groups full of forwarding bots,
devotional chant spam, and campaign accounts — *real* but catastrophically
**off-domain** for team-role archetypes. Ingesting them would have **increased**
the train→team domain shift that is already the model's main limitation. (The
extractor `archive/exploratory_scripts/parse_garimella.py` is kept for reproducibility; the data is
deliberately **not** wired into `CLEAN_FILES`.)

The key insight: *realness is not the only axis — domain match matters more.* Domain fit motivated the synthetic examples; this does not establish a general advantage over real data.

## What we did
`src/make_synthetic.py` generates **9 naturalistic small-team project chats**
(CS group project, hackathon, charity event, study group, startup, newsletter,
wellness circle, docs sprint, moderators) in English — the language of our own
team chat — totalling **268 messages / 53 users**. They were written to read like
real teammates coordinating a project, **not** as archetype caricatures; the role
only emerges from how each person communicates. The later teams are deliberately
weighted toward the starved **Butterfly** and **Capybara** roles.

This brings the labeled training set to **300 users** (183 Slack + 55 Nankani +
9 Mbada + 53 synthetic), with a much rounder class balance:
Bee 104 · Ant 73 · Low-reciprocity 56 · Capybara 35 · Butterfly 32.

Labels (`src/label_synthetic.py`, `dataset="synthetic"` in `opus_labels.json`)
are fixed soft-label distributions. Earlier notes describe reading messages rather than using persona comments, but the exact human versus LLM-assisted authoring process is not documented. All are `confidence="medium"`
(weight 0.7) so the synthetic set never outweighs the real labels.

### Validity guardrails
1. **Disclosed** — tagged `synthetic` everywhere, never silently mixed in.
2. **Ablation** — `python src/train_ml.py --no-synthetic` reproduces the no-aug
   model; both arms are reported below.
3. **Honest scorecard untouched** — the blind held-out **team** test
   (`wa_holdout_opus.json`) is 100 % real and never sees synthetic data.

## The headline metric: soft accuracy
On a fuzzy, overlapping target, argmax accuracy is misleading (a 27 %-bee /
26 %-ant tie scored as a full miss). We therefore report **soft accuracy =
mean Σ min(model, Opus)** — the probability mass the prediction shares with the
Opus soft label, in [0, 1]. It credits partial agreement on the whole
distribution. (Implemented in `train_ml.soft_metrics` and `compare_holdout`.)

## Results (ablation, 300 vs 247 users)

### Out-of-fold per-class F1 (internal CV)
| class | without synth | with synth |
|---|---|---|
| bee | 0.514 | **0.562** |
| ant | 0.446 | **0.463** |
| butterfly | 0.000 | **0.373** |
| capybara | 0.008 | **0.629** |
| leech | 0.124 | **0.418** |
| **macro-F1** | **0.273** | **0.559** |
| OOF soft-accuracy | 0.849 | 0.834 |
| CV fusion weight (metric) | 0.25 | 0.10 |

## Interpretation limits

These are upstream historical reported ablation figures, not a reproduced experiment. The reported class scores do not average to the reported 0.559 macro-F1; fold-level provenance remains unresolved. Do not claim these as a validated new gain. Small-team and individual outcomes are omitted from this edition.
