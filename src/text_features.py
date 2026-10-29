"""
text_features.py — scale-invariant TEXT features for COIN archetype distillation.

Two feature blocks are available:
  1. STYLE: 17 per-user lexical rates/means, used by the fusion model.
  2. EMBEDDING: mean 384-dimensional MiniLM vectors, exploratory only.

Per-message rates reduce dependence on message volume but do not establish
invariance across languages, group sizes or domains. Lexicon references justify
the measurements, not the validity of the five role categories. Only the most
recent 150 messages are selected per user; the full research backends are
separate from the lightweight offline demo. Embedding caches include selected
message content and schema in their signature and never reuse training cache
entries for a private team.
"""
import json
import re
import hashlib
import numpy as np
import config

from model_revisions import EMBEDDING_MODEL, EMBEDDING_REVISION

EMB_MODEL = EMBEDDING_MODEL          # 384-d, fast, strong general encoder
MAX_MSGS_PER_USER = 150                  # same depth Opus read
CACHE = config.TEXT_CACHE

CLEAN_FILES = dict(config.CLEAN_FILES)   # {"slack":…, "nankani":…, "mbada":…}

# ════════════════════════════════════════════════════════════════════════════
# Scientifically-grounded text-style features (v2)
# ════════════════════════════════════════════════════════════════════════════
# These sources motivate lexical measurements; they do not validate an
# archetype assignment for any individual:
#
#   • Empath categories — Fast, Chen & Bernstein (2016), CHI. 200 crowd-validated
#     categories, correlated r=0.906 with LIWC. Used for help/affection/
#     positive&negative emotion/work/achievement/communication/leader/giving/
#     politeness.  (LIWC tradition: Tausczik & Pennebaker 2010.)
#   • VADER compound sentiment — Hutto & Gilbert (2014), ICWSM. Social-media-tuned
#     valence, robust to emoji/slang.
#   • Speech-act signals — Searle (1976): question rate (directive) and
#     commitment rate (commissive "I'll / I can / on it").
#   • Structural pragmatics: @-mention, URL/link-share (the Butterfly info-bridge
#     signal), emoji rate, message length.
#
# Rates/means reduce message-volume dependence; domain shift can remain.

# ── lazy singletons (heavy-ish to construct, build once) ─────────────────────
_EMPATH = None
_VADER = None
EMPATH_CATS = ["help", "affection", "positive_emotion", "negative_emotion",
               "work", "achievement", "communication", "leader", "giving",
               "politeness"]


def _empath():
    global _EMPATH
    if _EMPATH is None:
        from empath import Empath
        _EMPATH = Empath()
    return _EMPATH


def _vader():
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
    return _VADER


# structural / speech-act regexes
_Q          = "?"
from features_v2 import _raw_mentions
_URL        = re.compile(r"https?://|www\.")
_EMOJI      = re.compile("[\U0001F000-\U0001FAFF☀-➿]")
_COMMIT     = re.compile(r"\b(i can|i'll|i will|let me|happy to|i could|on it|"
                         r"i'll do|i can do|i'll take|i can take|i'll handle|i got)\b", re.I)


def _style_vector(msgs):
    """17 scale-invariant, citable text-style signals for one user.

    Order matches STYLE_NAMES. Empath rates come from one normalized pass over
    the concatenated messages; the rest are per-message rates / means."""
    n = max(len(msgs), 1)
    lex = _empath()

    # Empath (Fast et al. 2016) — PRESENCE RATE: fraction of the user's messages
    # that express each category. More robust / less zero-inflated than
    # word-normalizing over all text, and directly interpretable ("X% of messages
    # express help / affection / …").
    emp_counts = [0] * len(EMPATH_CATS)
    for m in msgs:
        a = lex.analyze(m or " ", categories=EMPATH_CATS, normalize=False) or {}
        for i, c in enumerate(EMPATH_CATS):
            if (a.get(c, 0.0) or 0.0) > 0:
                emp_counts[i] += 1
    emp_vec = [c / n for c in emp_counts]

    # VADER (Hutto & Gilbert 2014) — mean compound sentiment over messages
    va = _vader()
    vader_mean = float(np.mean([va.polarity_scores(m)["compound"] for m in msgs])) if msgs else 0.0

    # speech-act + structural pragmatics
    q_rate       = sum(_Q in m for m in msgs) / n
    mention_rate = sum(bool(_raw_mentions(m)) for m in msgs) / n
    url_rate     = sum(bool(_URL.search(m)) for m in msgs) / n
    emoji_rate   = sum(len(_EMOJI.findall(m)) for m in msgs) / n
    commit_rate  = sum(bool(_COMMIT.search(m)) for m in msgs) / n
    avg_len      = float(np.mean([len(m.split()) for m in msgs])) / 20.0 if msgs else 0.0

    return np.array(emp_vec + [vader_mean, q_rate, mention_rate, url_rate,
                               emoji_rate, commit_rate, avg_len], dtype=np.float32)


STYLE_NAMES = [f"emp_{c}" for c in EMPATH_CATS] + [
    "vader_compound", "q_rate", "mention_rate", "url_rate",
    "emoji_rate", "commit_rate", "avg_len_norm"]


def _user_messages(dataset):
    """{author: [bodies]} from a clean file, text messages only."""
    out = {}
    for line in open(CLEAN_FILES[dataset], encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        body = str(r.get("body") or "").replace("\n", " ").strip()
        if not body or body == "nan":
            continue
        out.setdefault(r["author"], []).append(body)
    return out


def _team_messages(path=config.WHATSAPP_CLEAN):
    import features
    df = features.load_clean(path)
    out = {}
    for _, r in df.sort_values("datetime").iterrows():
        body = str(r["body"]).replace("\n", " ").strip()
        if not body or body == "nan":
            continue
        out.setdefault(r["author"], []).append(body)
    return out


def _signature(user_keys, messages):
    """Invalidate cached vectors when selected text, order or feature schema changes."""
    payload = {"version": 3, "model": EMB_MODEL, "revision": EMBEDDING_REVISION, "limit": MAX_MSGS_PER_USER,
               "style_names": STYLE_NAMES, "users": user_keys, "messages": messages}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                    sort_keys=True).encode()).hexdigest()


def build_features(user_index, team=False, use_cache=True, style_only=False):
    """
    user_index: list of (dataset, user_id) to build features for, in order.
    Returns (EMB [n×384], STYLE [n×17]).  Caches embeddings keyed by signature.

    style_only=True returns (None, STYLE) WITHOUT loading the sentence
    transformer at all. The production late-fusion model uses only the 17-d
    lexical style block, so this avoids loading a second torch model stack in the
    same process as ml_features' RoBERTa — which segfaults (exit 139) on macOS —
    and is much faster.
    """
    if not user_index:
        return (None if style_only else np.empty((0, 384), dtype=np.float32),
                np.empty((0, len(STYLE_NAMES)), dtype=np.float32))
    if style_only:
        if team:
            msg_map = {("team", a): m for a, m in _team_messages().items()}
            getter = lambda d, u: msg_map.get((d, u), [])
        else:
            caches = {}
            def getter(d, u):
                if d not in caches:
                    caches[d] = _user_messages(d)
                return caches[d].get(u, [])
        style = np.vstack([_style_vector(getter(d, u)[-MAX_MSGS_PER_USER:])
                           for d, u in user_index])
        return None, style

    # gather messages
    if team:
        msg_map = {("team", a): m for a, m in _team_messages().items()}
        getter = lambda d, u: msg_map.get((d, u), [])
    else:
        caches = {}
        def getter(d, u):
            if d not in caches:
                caches[d] = _user_messages(d)
            return caches[d].get(u, [])

    selected = [getter(d, u)[-MAX_MSGS_PER_USER:] for d, u in user_index]
    sig = _signature([f"{d}/{u}" for d, u in user_index], selected)
    # Private team data never read or write the training embedding cache.
    if use_cache and not team:
        try:
            with np.load(CACHE, allow_pickle=False) as cache:
                if str(cache["sig"]) == sig:
                    return cache["emb"], cache["style"]
        except (FileNotFoundError, KeyError, ValueError):
            pass

    from sentence_transformers import SentenceTransformer
    # Force CPU: Apple MPS intermittently segfaults this model (exit 139). CPU is
    # plenty fast for our ≤300 users and fully deterministic. Override with
    # SOCIALCOMPASS_DEVICE=mps if you really want the GPU path.
    import os
    _device = os.environ.get("SOCIALCOMPASS_DEVICE", "cpu")
    model = SentenceTransformer(EMB_MODEL, revision=EMBEDDING_REVISION, trust_remote_code=False, device=_device)

    emb_rows, style_rows = [], []
    flat_msgs, owners = [], []
    for i, msgs in enumerate(selected):
        style_rows.append(_style_vector(msgs))
        for m in msgs:
            flat_msgs.append(m)
            owners.append(i)

    # one batched encode pass over all messages, then mean-pool per user
    print(f"  embedding {len(flat_msgs)} messages from {len(user_index)} users …")
    vecs = (model.encode(flat_msgs, batch_size=256, show_progress_bar=False,
                         normalize_embeddings=True) if flat_msgs
            else np.empty((0, 384), dtype=np.float32))
    dim = vecs.shape[1] if len(vecs) else 384
    sums = np.zeros((len(user_index), dim), dtype=np.float32)
    cnts = np.zeros(len(user_index), dtype=np.float32)
    for vec, o in zip(vecs, owners):
        sums[o] += vec
        cnts[o] += 1
    emb = sums / np.maximum(cnts[:, None], 1.0)
    style = np.vstack(style_rows)

    if use_cache and not team:
        np.savez(CACHE, sig=sig, emb=emb, style=style)
    return emb, style


if __name__ == "__main__":
    # smoke: build for the 197 labeled users
    labels = [json.loads(l) for l in open(config.OPUS_LABELS) if l.strip()]
    idx = [(l["dataset"], l["user_id"]) for l in labels]
    emb, style = build_features(idx)
    print("emb", emb.shape, "style", style.shape)
    print("style means:", dict(zip(STYLE_NAMES, np.round(style.mean(0), 3))))
