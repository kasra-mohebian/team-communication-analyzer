# Behaviour features and inferred interactions

The model uses 13 behaviour features. `features.py` supplies the base measurements; `features_v2.py` replaces five interaction columns: `in_out_ratio`, `replies_sent`, `avg_mentions`, `betweenness_centrality` and `response_consistency`.

## How an edge is inferred

For each time-ordered message, explicit mentions take precedence. If there is no resolved mention, the fallback selects the most recent different author in a 45-minute window. The fallback window is separate for each available channel, conversation and team context.

Mention parsing supports Slack `<@handle>`, wrapped WhatsApp names, ordinary handles and the demo's `@Member 12` format. Overlapping parser matches count once. Exact author matches take precedence; ambiguous abbreviated names are not resolved by choosing the first person.

Edges are **inferred communication links**, not proof that a person replied to a particular message. The temporal heuristic, missing threads, ambiguous names and uncertain timestamps can all affect the network.

## What the five measurements mean

| Feature | Implementation |
|---|---|
| `in_out_ratio` | Inferred incoming reply acts divided by outgoing reply messages, capped at 50; when there are no outgoing replies, uses capped incoming count |
| `replies_sent` | Fraction of messages assigned at least one inferred target |
| `avg_mentions` | Mean number of parsed mention occurrences per message; unresolved mentions still count |
| `betweenness_centrality` | Normalised, unweighted betweenness of the inferred directed graph |
| `response_consistency` | Negative burstiness of inferred reply latencies; insufficient observations become 0 |

The research dashboard uses the same v2 graph builder as the feature extractor. Edge weights record interaction counts; shortest-path betweenness does not interpret those counts as distances.

## Scope and limitations

The other eight measurements include lexical content, message length, MATTR and initiation rate. Current base timing and optional topic-shift comparisons also respect conversation context. This is a further change from the historical feature implementation.

Mbada timestamps are an artificial ordering ladder; timing measurements there do not represent real delays. Rates do not remove all group-size or language effects. Network measurements, weak labels and communication roles are exploratory proxies, not validated measures of personality or work quality.

The historical scores in RESULTS.md were not recomputed with these changes. Running `python src/build_v2_features.py` requires local base feature matrices and clean training streams; a private case-study stream is optional for that preparation step.
