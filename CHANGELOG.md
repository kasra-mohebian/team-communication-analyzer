# Changes in the portfolio edition

## 2026-10-27

- Retained the upstream analysis implementation; excluded individual records, embedded individual examples and original Git history. Published only selected training-wide summaries with numerical values unchanged.
- Added an offline feature/network demo and a separate optional demo dashboard with numbered fictional members.
- Added the missing base-feature builder and documented the separate labelling prerequisites.
- Replaced broad system-word filtering with conservative notice detection; exact duplicate messages are removed in new WhatsApp/Nankani parses. Parser imports no longer read private exports.
- Reject negative, out-of-range, nonfinite and boolean teacher probabilities.
- Isolate inferred replies by available channel/conversation/team keys and use the same v2 graph in the research dashboard. Accept timezone-aware timestamps.
- Compute future Fusion per-class scores from Fusion predictions, retaining separate single-view scores. Reject invalid fusion weights.
- Align future style windows with the most recent messages and reuse mention parsing that excludes email fragments. Remove pickle-enabled optional cache loading.
- Audit existing destination files rather than returning success when private inputs are absent. Exporting requires a real local map, complete core artifacts and a fresh destination. New exports default to ignored outputs/runs/public.
- Escape member strings in research-dashboard HTML and describe leading-role share without claiming calibrated confidence.
- Add regression tests and a GitHub Actions workflow with pinned official actions.

These changes alter future feature values and predictions. No historical metric is recomputed, overwritten or advertised as a new performance gain. Full cross-validation/training and all optional neural backends have not been rerun.

## External-reader review

- Rewrote the README around an executable quick start, expected outputs, model explanation and clearly separated historical evidence. Updated the research and feature guides.
- Fixed numbered/overlapping mention parsing and ambiguous author-prefix resolution.
- Scoped base initiation/topic timing and Slack conversation flags to the correct context; added Slack import safety, epoch-date handling and deduplication.
- Preserved ordinary Mbada prose containing words such as null; clarified artificial timestamp limitations.
- Validated labels at training/annotation entry points and rejected duplicate units.
- Included message content and schema in embedding-cache signatures; excluded team/training cache reuse and handled empty text-feature input.
- Limited automatically trackable public outputs to the reviewed aggregate filenames and kept generated annotation/report material ignored.
- Added regression cases for these failures. Historical aggregate values are unchanged; full research retraining was not rerun.
