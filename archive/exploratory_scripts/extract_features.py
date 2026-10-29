"""
extract_features.py
═══════════════════
Pipeline step: extract the 13 active metrics for the team's own WhatsApp chat
and save them for inspection. The heavy lifting lives in `features.py` (shared
with the ML layer) so this file is a thin, sanity-checked runner.

    whatsapp_clean.json  →  features.extract_features  →  wa_features.json

The 13 metrics and their scientific sources are documented in features.py.
Run `python parse_whatsapp.py` first to produce whatsapp_clean.json.
"""

from features import (
    ALL_METRICS, ABSOLUTE_METRICS, RANK_METRICS,
    load_clean, extract_features, sanity_check_features,
)

import config
INPUT_FILE  = config.WHATSAPP_CLEAN
OUTPUT_FILE = config.WA_FEATURES

if __name__ == "__main__":
    print("\n🔎  SocialCompass — Feature extraction (13 active metrics)")
    print("=" * 64)

    df = load_clean(INPUT_FILE)
    feats = extract_features(df, label="Team WhatsApp")

    # Required sanity layer: distributions + range asserts + near-zero variance.
    flagged = sanity_check_features(feats, label="Team WhatsApp")

    print(f"\n  Absolute-scored metrics ({len(ABSOLUTE_METRICS)}): {ABSOLUTE_METRICS}")
    print(f"  Rank-scored metrics    ({len(RANK_METRICS)}): {RANK_METRICS}")

    print("\n── Full feature table ── " + "─" * 40)
    print(feats.set_index("author")[ALL_METRICS].round(4).to_string())

    feats.to_json(OUTPUT_FILE, orient="records", lines=True)
    print(f"\n  ✅ Feature matrix {feats.shape} → {OUTPUT_FILE}")
    if flagged:
        print(f"  ⚠ Note for the paper: {flagged} show near-zero variance on this "
              f"5-person group (expected for betweenness on a tiny network).")
