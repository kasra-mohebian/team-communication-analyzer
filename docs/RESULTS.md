# Historical evaluation and limits

The public JSON files contain selected training-wide summaries from the upstream course snapshot. Numerical values retained are unchanged. Case-study, domain-shift and individual records have been omitted for privacy; this is a disclosed subset, not the complete original experiment.

## Recorded model comparison

The repeated-CV macro-F1 values are 0.1001 for the mean baseline, 0.4338 for metric Random Forest, 0.4889 for metric XGBoost and 0.5114 for Fusion. The historical per_class_f1 table belongs to XGBoost: its five scores average to 0.4889128. Do not present that table as Fusion class scores. Corrected future training records separate Fusion and single-view per-class reports.

The stored blend weight was manually set to 0.50; the recorded CV sweep favoured about 0.15. The actual selected settings and stored evaluation are not presented as an independently optimised or newly reproduced result.

## Human agreement

The training-study aggregate records include 50 commonly annotated users. Human–human distribution overlap was 0.6928, hard agreement 0.40 and Cohen kappa 0.2264. These are stored summaries; the per-user annotation tables and teacher answer key are not distributed. No agreement recomputation is claimed from the public edition.

## Interpretation

- 247 external users and 53 fictional users with fixed soft-label distributions (human versus LLM-assisted authoring is not documented) made up the 300 labels. Synthetic examples particularly supported the smaller classes.
- Labels are weak supervision from the same underlying messages, not independent psychological ground truth.
- CV divides users within existing communities; networks/ranks were computed before splitting. This is transductive evaluation. Fusion-weight tuning and evaluation also reuse overlapping folds, so selection optimism can remain. Group-held-out nested validation remains future work.
- Language lexicons, inferred replies, uncertain label agreement and domain shift limit generalisation.
- In-sample and cross-validation scores are different quantities. Corrected code has not been fully retrained here, and no accuracy improvement is claimed.

The private submitted report remains unchanged. Future experiments need their own authorised inputs, feature versions, settings, fold predictions and fresh output directory.

## Label inputs and review revision

The earlier label formatter supplied the author identifier, total message count and `[R]`/`[I]` reply/initiation tags alongside message bodies. These overlap with behavioural features, so the historical labels were not independent of those signals. The revised formatter omits that header metadata and uses a fixed pseudonymous participant label. It has not been used to regenerate labels or scores. Text and sampling still carry behavioural signals; this change cannot establish independent ground truth.

All model means and fold standard deviations are now shown in the README. Comparing standard deviations is not a significance test; no paired statistical advantage is claimed. No stored JSON value was edited.
