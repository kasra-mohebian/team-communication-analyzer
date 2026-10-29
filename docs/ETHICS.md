# Privacy and publication scope

This edition publishes source code, wholly fictional demonstrations and selected training-wide numerical summaries. It excludes real chat exports, cleaned transcripts, identity maps, individual role distributions, per-user features, real interaction graphs, per-user annotation CSVs, small-team case-study files and original course reports/slides. The original upstream Git history is not included.

Replacing a name with a letter or number does not make a behavioural profile anonymous. For that reason those profiles are omitted rather than perturbed or relabelled. Historical numbers retained in the training-wide summaries have not been changed. Exclusions are documented in UPSTREAM.md; omitted results are not claimed to be reproducible from this edition.

The offline demo uses invented text and fictional numbered participants. It performs feature extraction, not a validated personality assessment. Role labels are exploratory communication categories, not diagnoses, employee ratings or measures of contribution quality.

Research scripts remain capable of processing private local inputs. Their generated files are not part of the publication set. The teacher-labelling script sends excerpts to an external API when explicitly run; it is not used by the offline demo. Only use inputs you are authorised to process for the actual purpose and service. A software licence does not provide permission to disclose personal data or third-party datasets.

The public-artifact checker rejects non-allowlisted files and individual-record fields in outputs/public. It is a targeted technical safeguard, not a legal certification. Original copies on other people's accounts are outside this edition's control.

## External label processing

The pre-review formatter inserted its `user_id` argument verbatim, with message totals and inferred reply markers. Some source parsers retain display names; therefore that code path could disclose a name to the external service. The public files alone do not establish which identifiers were sent in every historical request. Historical API requests have not been audited here, so this edition does not certify their anonymisation.

The revised formatter omits the identifier and explicit behavioural metadata and uses `U0001` as a per-prompt participant label. This does **not** redact message bodies: bodies can contain names, phone numbers, quotations or other identifying content. Review and redact the actual text before any authorised external processing. API execution now requires an explicit `--allow-external-api` flag; neither CI nor the fictional demo uses the API. Local identifiers/output files remain private.

Only the dependency-light offline environment is tested in CI. The pinned full research stack has not been installed or retrained as part of this review.
