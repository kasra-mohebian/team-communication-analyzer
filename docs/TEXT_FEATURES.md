# The text-style feature view

The research model uses 17 lexical rates and averages. Empath supplies ten message-presence categories; VADER supplies mean sentiment; regular expressions provide the remaining structural signals. These are measurable proxies, not validated role or personality tests.

| # | Feature | What it measures | Exploratory association | Source |
|---|---|---|---|---|
| 1 | `emp_help` | messages expressing helping | 🦫/🐜 | Empath (Fast 2016) |
| 2 | `emp_affection` | warmth / care | 🦫 Capybara | Empath |
| 3 | `emp_positive_emotion` | positive affect | 🦫 Capybara | Empath |
| 4 | `emp_negative_emotion` | negative affect | (contrast) | Empath |
| 5 | `emp_work` | task / work language | 🐜 Ant | Empath |
| 6 | `emp_achievement` | completion / achievement | 🐜 Ant | Empath |
| 7 | `emp_communication` | communicative / social verbs | 🐝 Bee | Empath |
| 8 | `emp_leader` | leadership / directing | 🐝 Bee | Empath |
| 9 | `emp_giving` | sharing / giving | 🦋 Butterfly | Empath |
| 10 | `emp_politeness` | polite forms | 🦫 Capybara | Empath; Danescu (2013) |
| 11 | `vader_compound` | mean sentiment valence | 🦫 Capybara | VADER (Hutto 2014) |
| 12 | `q_rate` | question rate (directive act) | 🐝 Bee | Searle (1976) |
| 13 | `mention_rate` | @-addressing others | 🐝 Bee | structural |
| 14 | `url_rate` | link / resource sharing | 🦋 Butterfly | structural (key bridge signal) |
| 15 | `emoji_rate` | emoji per message | affect | structural |
| 16 | `commit_rate` | commitments ("I'll / I can / on it") | 🐜/🦫 | commissive act, Searle (1976) |
| 17 | `avg_len_norm` | mean message length (÷20) | 🐜 Ant | structural |


The model uses the most recent 150 nonempty messages in clean-stream order per user. Message rates reduce volume dependence but do not remove language, group-size or domain effects. Text-style and network views distil labels from the same underlying messages, so they are not independent ground truths.

The optional embedding view averages MiniLM message vectors and is retained for exploratory scripts. The fusion model uses the 17-dimensional style view without loading that encoder. Cache signatures include selected message text, user order and feature schema; team features never reuse the training cache.

The lightweight public demo uses behaviour features with lexical fallbacks; it does not load Empath/VADER text-style training or a fitted classifier. See REPRODUCING.md for the separate research dependencies and RESULTS.md for historical evaluation limits.
