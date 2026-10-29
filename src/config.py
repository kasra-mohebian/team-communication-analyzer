"""
config.py — single source of truth for every data / output path.

Paths are resolved relative to the PROJECT ROOT via __file__, so scripts work
no matter what the current working directory is (run `python src/train_ml.py`
from the repo root, or from anywhere). Importing this module also creates the
output directories if they are missing.

Layout
------
  data/raw/        original exports (WhatsApp .txt, Nankani .txt, Mbada .docx, Slack)
  data/clean/      parsed one-JSON-per-line message streams
  data/features/   13-metric matrices (v1 and v2)
  data/labels/     Opus soft labels + blind held-out gold
  outputs/models/  fitted model + embedding cache
  outputs/results/ predictions, CV reports, evaluation JSONs
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # repo root (src/ -> ..)

DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
FEATURES = DATA / "features"
LABELS = DATA / "labels"

OUTPUTS = ROOT / "outputs"
MODELS = OUTPUTS / "models"
RESULTS = OUTPUTS / "results"
ANNOTATION = OUTPUTS / "annotation"   # human inter-rater validation study artifacts

for _d in (CLEAN, FEATURES, LABELS, MODELS, RESULTS, ANNOTATION):
    _d.mkdir(parents=True, exist_ok=True)

# ── raw ──────────────────────────────────────────────────────────────────────
NANKANI_RAW = str(RAW / "Nankani_2020.txt")
WHATSAPP_RAW = str(RAW / "whatsapp_groupchat.txt")
SLACK_RAW_DIR = str(RAW / "slack_data")
GARIMELLA_RAW = str(RAW / "Garimella_Kiran_15_6_2020_anonymized.csv.gz")

# ── clean message streams ────────────────────────────────────────────────────
SLACK_CLEAN = str(CLEAN / "slack_clean.json")
NANKANI_CLEAN = str(CLEAN / "nankani_clean.json")
MBADA_CLEAN = str(CLEAN / "mbada_clean.json")
GARIMELLA_CLEAN = str(CLEAN / "garimella_clean.json")
SYNTHETIC_CLEAN = str(CLEAN / "synthetic_clean.json")
WHATSAPP_CLEAN = str(CLEAN / "whatsapp_clean.json")
# NOTE: garimella is intentionally NOT in CLEAN_FILES — the dataset is broadcast/
# spam/propaganda, off-domain for team archetypes (see docs note). Kept available
# for the extractor only, not wired into the training pipeline.
# synthetic IS included: disclosed in-domain augmentation for the rare classes
# (see make_synthetic.py); train_ml can ablate it out via --no-synthetic.
CLEAN_FILES = {"slack": SLACK_CLEAN, "nankani": NANKANI_CLEAN,
               "mbada": MBADA_CLEAN, "synthetic": SYNTHETIC_CLEAN}

# ── feature matrices (13 metrics) ────────────────────────────────────────────
TRAIN_FEATURES = str(FEATURES / "training_results.json")        # v1
TRAIN_FEATURES_V2 = str(FEATURES / "training_results_v2.json")  # v2 (production)
WA_FEATURES = str(FEATURES / "wa_features.json")                # v1 team
WA_FEATURES_V2 = str(FEATURES / "wa_features_v2.json")          # v2 team (production)

# ── labels ───────────────────────────────────────────────────────────────────
OPUS_LABELS = str(LABELS / "opus_labels.json")
WA_HOLDOUT = str(LABELS / "wa_holdout_opus.json")

# ── outputs: models ──────────────────────────────────────────────────────────
ML_MODEL = str(MODELS / "ml_model.joblib")
TEXT_CACHE = str(MODELS / "text_cache.npz")

# ── outputs: results ─────────────────────────────────────────────────────────
WA_RESULTS = str(RESULTS / "wa_results.json")
ML_RESULTS = str(RESULTS / "ml_results.json")
WA_RULE_RESULTS = str(RESULTS / "wa_rule_results.json")
EVAL_VS_OPUS = str(RESULTS / "eval_vs_opus.json")
OPTIMIZE_RESULTS = str(RESULTS / "optimize_results.json")
AB_RESULTS = str(RESULTS / "ab_features_results.json")
HOLDOUT_COMPARISON = str(RESULTS / "holdout_comparison.json")
INSAMPLE_CHECK = str(RESULTS / "insample_check.json")
SYNTHETIC_ABLATION = str(RESULTS / "synthetic_ablation.json")

# ── outputs: human inter-rater validation study ──────────────────────────────
ANNOTATION_SAMPLE = str(ANNOTATION / "sample_users.json")            # blinded, shared
ANNOTATION_ANSWER_KEY = str(ANNOTATION / "answer_key_DO_NOT_SHARE.json")  # Opus labels, kept hidden
ANNOTATION_ASSIGNMENT = str(ANNOTATION / "assignment.json")          # study design metadata
ANNOTATION_RETURNED = str(ANNOTATION / "returned")                   # filled CSVs come back here
ANNOTATION_AGREEMENT = str(RESULTS / "annotation_agreement.json")    # computed IAA report

# ── outputs: selected training-wide public summaries ────────────────────────
# Individual results stay local; public exports contain aggregate fields only.
HISTORICAL_PUBLIC = OUTPUTS / "public"
PUBLIC = OUTPUTS / "runs" / "public"
# Export directories are created only after complete input validation.

# ── team identity (PRIVATE — never committed) ────────────────────────────────
CONFIG_DIR = ROOT / "config"
TEAM_CONFIG = CONFIG_DIR / "team.json"
TEAM_CONFIG_EXAMPLE = CONFIG_DIR / "team.example.json"


def load_team():
    """
    Load the private team map (config/team.json).

    This private configuration maps display names. Raw/clean corpora and derived
    research outputs may also contain identities and must stay local. Falls back to the
    committed template so a fresh clone still imports and runs — the parser will
    then simply match nobody and keep authors as they appear in the export.

    Returns a dict with:
      members              [{match, key, alias}, ...]  in display order
      group_author_markers [str, ...]                  group/system pseudo-authors
      keys                 [key, ...]
      alias_of             {key: alias}
    """
    path = TEAM_CONFIG if TEAM_CONFIG.exists() else TEAM_CONFIG_EXAMPLE
    cfg = json.loads(path.read_text(encoding="utf-8"))
    members = cfg.get("members", [])
    return {
        "source": str(path),
        "is_template": path == TEAM_CONFIG_EXAMPLE,
        "members": members,
        "group_author_markers": cfg.get("group_author_markers", []),
        "keys": [m["key"] for m in members],
        "alias_of": {m["key"]: m.get("alias", m["key"]) for m in members},
    }
