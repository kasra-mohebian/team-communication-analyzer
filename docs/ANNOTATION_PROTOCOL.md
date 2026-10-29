# Human annotation protocol

The original course study used two human raters and a common sample of 50 external-corpus users. Raters assigned a soft distribution across the five communication categories. The public edition retains aggregate agreement summaries, not individual annotations, reference keys or transcript-bearing annotation apps.

## Local study workflow

Only use messages that may be processed and shared with the intended raters for this purpose. Source availability does not by itself provide that permission.

1. Prepare local clean streams and validated weak labels.
2. Run `python src/build_annotation_sample.py` to build local annotation apps and a separate reference key.
3. Give each rater only their authorised app. The app embeds transcripts and remains private.
4. Each rater independently splits 100 points across the five categories, selects a confidence level and exports a CSV.
5. Run `python src/analyze_agreement.py --returned LOCAL_CSV_FOLDER --answer-key LOCAL_ANSWER_KEY`.

The app's presentation order differs by rater. It does not show the reference answers. Browser storage is local working storage, not the completed deliverable; raters must export their CSV.

## Input checks and interpretation

The analysis accepts approximately 100 points or a probability distribution summing approximately to one, normalises valid rows and rejects negative, non-finite, zero-total or duplicate rater/user records. Reference distributions are also validated.

Agreement summaries include distribution overlap, cosine similarity, Jensen–Shannon divergence, hard-label agreement and inter-rater coefficients. Agreement describes consistency among these raters; it does not establish independent psychological validity, prove the language model is correct, or demonstrate that it outperforms each human.

The original 50-user summaries are historical evidence. The complete study cannot be recomputed from this public edition because its per-user inputs are intentionally excluded. Differences in historical teacher/annotation text windows also limit interpretation; see RESULTS.md.
