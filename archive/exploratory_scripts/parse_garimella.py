"""
parse_garimella.py
──────────────────
Targeted miner for the Garimella et al. (2020) anonymized Indian-WhatsApp
public-groups dataset (6.3 M messages, 5.7 k groups, multilingual).

These are large political *broadcast* groups — very far from a small team chat —
so we do NOT bulk-ingest them. The goal is the opposite: pull a *small, clean,
balanced* candidate pool, biased toward (a) English-heavy users we can label
without translation noise and (b) the rare archetypes we are short on
(Butterfly via forwarders, Capybara via supportive/emoji users, plus Bee/Ant).

Two passes over the gzip stream:
  1. aggregate per (group, user): original-text count, forward count, English
     ratio, link / emoji / question / length signals  → pick a diverse pool.
  2. collect each picked user's actual messages → human-readable candidate dump
     (`outputs/results/garimella_candidates.json`) for Opus to read & label.

The clean training stream (`data/clean/garimella_clean.json`) is written later,
*only* for the users Opus actually labels, by `build_garimella_clean.py`.
"""
import gzip
import json
import re
import config

RAW = config.GARIMELLA_RAW
CAND_DUMP = str(config.RESULTS / "garimella_candidates.json")

MIN_ORIG      = 15      # min original (non-forwarded) text messages to qualify
MAX_PER_GROUP = 2       # diversity: at most N candidates from one group
POOL_TARGET   = 90      # how many candidates to dump for labeling
MAX_MSGS      = 32      # messages collected per candidate for reading

_URL   = re.compile(r"https?://")
_LATIN = re.compile(r"[A-Za-z]")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")


def _is_forward(f):
    return not (f == "0" or f == "" or f == "None")


def _english_ratio(text):
    """Share of alphabetic characters that are ASCII latin (English proxy)."""
    latin = len(_LATIN.findall(text))
    alpha = sum(c.isalpha() for c in text)
    return latin / alpha if alpha else 0.0


def pass1():
    """Aggregate per-(group,user) signals for qualifying users."""
    agg = {}   # key -> dict of running stats
    with gzip.open(RAW, "rt", encoding="utf-8", errors="replace") as fh:
        next(fh)  # header
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            gid, uid, text, _ts, _mt, _mu, fwd = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            key = (gid, uid)
            a = agg.get(key)
            if a is None:
                a = agg[key] = {"orig": 0, "fwd": 0, "latin": 0.0, "links": 0,
                                "emoji": 0, "q": 0, "chars": 0}
            if _is_forward(fwd):
                a["fwd"] += 1
            has_text = text and text != "None"
            if has_text and not _is_forward(fwd):
                a["orig"]  += 1
                a["latin"] += _english_ratio(text)
                a["chars"] += len(text)
                if _URL.search(text):   a["links"] += 1
                if "?" in text:         a["q"]     += 1
                if _EMOJI.search(text): a["emoji"] += 1
    # keep only qualifying users and finalize ratios
    out = {}
    for key, a in agg.items():
        if a["orig"] < MIN_ORIG:
            continue
        o = a["orig"]; tot = o + a["fwd"]
        out[key] = {
            "orig": o, "fwd": a["fwd"],
            "english":  round(a["latin"] / o, 3),
            "fwd_rate": round(a["fwd"] / tot, 3) if tot else 0.0,
            "link_rate":  round(a["links"] / o, 3),
            "emoji_rate": round(a["emoji"] / o, 3),
            "q_rate":     round(a["q"] / o, 3),
            "avg_len":    round(a["chars"] / o, 1),
        }
    return out


def pick_pool(stats):
    """Choose a diverse, English-heavy candidate pool spanning archetype hints."""
    eng = {k: s for k, s in stats.items() if s["english"] >= 0.6}

    def bucket(s):
        if s["fwd_rate"] >= 0.45 and s["link_rate"] >= 0.05: return "butterfly?"
        if s["emoji_rate"] >= 0.25 and s["avg_len"] <= 60:   return "capybara?"
        if s["q_rate"] >= 0.18 and s["orig"] >= 40:          return "bee?"
        if s["avg_len"] >= 120 and s["emoji_rate"] < 0.1:    return "ant?"
        if s["orig"] <= 25 and s["fwd_rate"] >= 0.4:         return "leech?"
        return "mixed"

    # group-cap for diversity, then round-robin across buckets toward POOL_TARGET
    from collections import defaultdict
    per_group = defaultdict(int)
    buckets = defaultdict(list)
    for k, s in sorted(eng.items(), key=lambda kv: -kv[1]["orig"]):
        gid = k[0]
        if per_group[gid] >= MAX_PER_GROUP:
            continue
        per_group[gid] += 1
        buckets[bucket(s)].append((k, s))

    pool, order = [], ["capybara?", "butterfly?", "leech?", "ant?", "bee?", "mixed"]
    i = 0
    while len(pool) < POOL_TARGET and any(buckets[b] for b in order):
        b = order[i % len(order)]
        if buckets[b]:
            pool.append(buckets[b].pop(0) + (b,))
        i += 1
    return pool


def pass2(pool):
    """Collect each picked user's messages for human reading/labeling."""
    want = {(k[0], k[1]): {"hint": hint, "stats": s, "msgs": []}
            for k, s, hint in pool}
    with gzip.open(RAW, "rt", encoding="utf-8", errors="replace") as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            gid, uid, text, fwd = p[0], p[1], p[2], p[6]
            key = (gid, uid)
            rec = want.get(key)
            if rec is None or _is_forward(fwd) or not text or text == "None":
                continue
            if len(rec["msgs"]) < MAX_MSGS:
                rec["msgs"].append(text)
    # serialize with readable composite ids
    dump = []
    for (gid, uid), rec in want.items():
        dump.append({
            "garimella_id": f"g{gid}_u{uid}",
            "group": gid, "user": uid,
            "hint": rec["hint"], "stats": rec["stats"],
            "messages": rec["msgs"],
        })
    dump.sort(key=lambda d: d["hint"])
    return dump


if __name__ == "__main__":
    print("Pass 1: aggregating qualifying users …")
    stats = pass1()
    print(f"  qualifying users (>= {MIN_ORIG} original text msgs): {len(stats):,}")
    pool = pick_pool(stats)
    from collections import Counter
    print("  pool by hint:", dict(Counter(h for _, _, h in pool)))
    print("Pass 2: collecting messages for", len(pool), "candidates …")
    dump = pass2(pool)
    with open(CAND_DUMP, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1)
    print(f"✅ wrote {len(dump)} candidates → {CAND_DUMP}")
