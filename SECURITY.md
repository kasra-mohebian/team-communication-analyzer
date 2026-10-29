# Security and private data

Run the fictional offline demo with requirements-demo.txt. The optional neural research stack downloads models and requires separately authorised data; it is not a production service or an exact historical environment. Keep real messages, identity maps, API credentials, individual predictions and generated environment logs out of public issues and commits.

External labelling requires explicit --allow-external-api; that flag does not establish consent or redact message text. Only load trusted model files; joblib/pickle deserialization can execute code. For a suspected credential or private-data exposure, do not post the value publicly. Use GitHub private vulnerability reporting when available; otherwise report only a non-sensitive description.

Combined annotation statistics are published; per-rater statistics are excluded by the exporter. The earlier repository history is held in a private archive; this public edition starts from the minimized summary. Previously downloaded copies are outside this repository's control.

## Research dependency review (2026-10-29)

The reviewed direct pins use NLTK 3.10.3, sentence-transformers 5.6.0 and torch 2.13.0. This resolves the earlier identified pin ranges for CVE-2026-12243, CVE-2026-68770 and CVE-2025-3000, but is not a clean bill of health for all dependencies. An audit still flags NLTK advisory [GHSA-8mgp-746c-j5xp](https://github.com/advisories/GHSA-8mgp-746c-j5xp), with no fixed version listed: model-artifact paths can bypass NLTK's path-security helpers. This repository does not call the affected persistence APIs or accept user-selected NLTK model paths. Do not treat NLTK pathsec as a containment boundary, and do not load untrusted model artifacts. The offline demo does not require NLTK. A full transitive research-stack installation and neural run remain unvalidated.
