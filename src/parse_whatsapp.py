import pandas as pd
import re
import config
from message_cleaning import is_system_body

# ── Load your exported WhatsApp .txt file ────────────────────────────────
WHATSAPP_FILE = config.WHATSAPP_RAW

# ── Map raw WhatsApp author labels → clean internal keys ─────────────────
# Real identities live ONLY in config/team.json, which is git-ignored. See
# config/team.example.json for the format. A WhatsApp export shows whatever
# display name the exporting phone has stored ("~ A. Nother", "Jane Uni
# Bamberg", a bare phone number), so we match a lower-cased SUBSTRING of the
# author field rather than an exact string.
_TEAM = config.load_team()
NAME_SUBSTR_MAP = {m['match']: m['key'] for m in _TEAM['members']}

# Authors that are really the GROUP itself / system, not a person → dropped.
# (WhatsApp posts "You created this group", encryption notice, etc. under the
#  group's own name.)
GROUP_AUTHOR_MARKERS = _TEAM['group_author_markers']

if _TEAM['is_template']:
    print("⚠  config/team.json not found — using the committed template. "
          "Authors will be kept exactly as they appear in the export.\n"
          "   cp config/team.example.json config/team.json  and fill it in.")


def normalize_author(raw):
    """Return a configured internal key, an unmatched label, or None for system authors."""
    a = raw.strip().lstrip('~').strip()          # drop WhatsApp "~ " non-contact prefix
    low = a.lower()
    if any(m in low for m in GROUP_AUTHOR_MARKERS):
        return None                              # the group pseudo-user → drop
    for match, key in NAME_SUBSTR_MAP.items():
        if match in low:
            return key
    return a                                     # unknown but real → keep as-is

def parse_whatsapp(filepath):
    """Parse WhatsApp export into a clean DataFrame."""

    pattern = re.compile(
        r'[‎\[]?'                                         # optional LTR mark or [
        r'(\d{1,2}[\/\.]\d{1,2}[\/\.]\d{2,4}),?\s'
        r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)'
        r'[\]\s]*[-–]?\s*'                                    # ] or space then optional dash
        r'([^:]+?):\s(.+)'
    )

    messages    = []
    current_msg = None

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line  = line.strip()
        match = pattern.match(line)

        if match:
            # Save previous message before starting new one
            if current_msg:
                messages.append(current_msg)

            date_str, time_str, author, body = match.groups()

            # ── Apply name mapping; skip the group/system pseudo-author ───
            author_clean = normalize_author(author)
            if author_clean is None:
                current_msg = None          # group/system line → ignore entirely
                continue

            current_msg = {
                'date_str': date_str,
                'time_str': time_str,
                'author'  : author_clean,
                'body'    : body.strip()
            }
        else:
            # Continuation of a multi-line message
            if current_msg and line:
                current_msg['body'] += ' ' + line

    # Don't forget the last message
    if current_msg:
        messages.append(current_msg)

    df = pd.DataFrame(messages)

    if len(df) == 0:
        print("❌ No messages parsed. Check your file format.")
        return df

    # ── Parse datetime ────────────────────────────────────────────────────
    df['datetime'] = pd.to_datetime(
        df['date_str'] + ' ' + df['time_str'],
        dayfirst=True, format='mixed',
        errors='coerce'
    )

    # Filter export notices without deleting prose containing 'left' or 'null'.
    df = df[~df['body'].map(is_system_body)]

    # ── Drop very short messages and bad rows ─────────────────────────────
    df = df[df['body'].str.len() >= 2]
    df = df.dropna(subset=['datetime'])
    df = df.drop_duplicates(['author', 'datetime', 'body'])
    df = df.sort_values('datetime', kind='stable').reset_index(drop=True)

    # ── Detect replies (within 10 mins from a different person) ──────────
    df['prev_time']   = df['datetime'].shift(1)
    df['prev_author'] = df['author'].shift(1)
    df['gap_mins']    = (
        df['datetime'] - df['prev_time']
    ).dt.total_seconds() / 60

    df['is_reply'] = (
        (df['gap_mins'] <= 10) &
        (df['prev_author'] != df['author'])
    ).astype(int)

    df['parent_id'] = df.apply(
        lambda r: f"t1_approx_{r.name - 1}"
                  if r['is_reply'] == 1
                  else f"t3_post_{r.name}",
        axis=1
    )

    # ── Final columns ─────────────────────────────────────────────────────
    df['subreddit'] = 'whatsapp_group'
    df['score']     = 1

    df = df[['author', 'datetime', 'body', 'parent_id',
             'is_reply', 'subreddit', 'score']]

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"✅ Parsed {len(df)} messages")
    print(f"👥 Unique users: {df['author'].nunique()}")
    print(f"📅 Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"↩️  Reply rate: {df['is_reply'].mean() * 100:.1f}%")
    print(f"\nUsers found:")
    print(df['author'].value_counts())
    print(f"\nSample:")
    print(df[['author', 'datetime', 'body', 'is_reply']].head(5))

    return df


def main():
    df = parse_whatsapp(WHATSAPP_FILE)
    if len(df) > 0:
        df.to_json(config.WHATSAPP_CLEAN, orient="records", lines=True)
        print(f"Saved to {config.WHATSAPP_CLEAN}")


if __name__ == "__main__":
    main()
