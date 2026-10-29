"""
parse_nankani.py
────────────────
Parses the Nankani_2020 WhatsApp export from the CODERS group at TSEC
(Thadomal Shahani Engineering College, Mumbai).

Format: DD/MM/YYYY, H:MM am/pm - Author: Message
~23,000 lines · Jan–Oct 2020 · Real software engineering students
discussing coding, GCP, GitHub, web development, hackathons.

This dataset is used as ADDITIONAL TRAINING DATA alongside the
Slack developer chat dataset.
"""

import pandas as pd
import re
import os
import config
from message_cleaning import is_system_body

INPUT_FILE = config.NANKANI_RAW
OUTPUT_FILE = config.NANKANI_CLEAN

# ── Pattern for this specific WhatsApp format ─────────────────────────────
# Format: 26/01/2020, 4:19 pm - Author Name: Message body
pattern = re.compile(
    r'(\d{1,2}/\d{1,2}/\d{4}),\s'          # date: DD/MM/YYYY
    r'(\d{1,2}:\d{2}\s?[aApP][mM])\s-\s'   # time: H:MM am/pm
    r'([^:]+?):\s'                           # author (up to first colon)
    r'(.+)'                                  # message body
)

# ── System message phrases to remove ──────────────────────────────────────
system_phrases = [
    'messages and calls are end-to-end encrypted',
    'joined using this group',
    'left', 'removed', 'added',
    'changed the subject',
    'changed this group',
    'you were added',
    '<media omitted>',
    'image omitted', 'video omitted',
    'audio omitted', 'document omitted',
    'sticker omitted', 'null',
    'created group', 'your security code',
    'missed voice call', 'missed video call'
]

def is_system_message(author, body):
    return is_system_body(body)

def parse_nankani(filepath):
    print(f"Reading {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Total lines: {len(lines):,}")

    messages    = []
    current_msg = None

    for line in lines:
        line  = line.strip()
        if not line:
            continue

        match = pattern.match(line)

        if match:
            # Save previous message
            if current_msg:
                messages.append(current_msg)

            date_str, time_str, author, body = match.groups()

            author = author.strip()
            body   = body.strip()

            # Skip system messages at parse time
            if is_system_message(author, body):
                current_msg = None
                continue

            current_msg = {
                'date_str': date_str,
                'time_str': time_str,
                'author'  : author,
                'body'    : body,
                'source'  : 'nankani_2020'
            }
        else:
            # Multi-line message continuation
            if current_msg and line:
                # Don't append if it looks like a system message continuation
                if not is_system_body(line):
                    current_msg['body'] += ' ' + line

    # Don't forget last message
    if current_msg:
        messages.append(current_msg)

    df = pd.DataFrame(messages)
    print(f"Raw messages parsed: {len(df):,}")

    if len(df) == 0:
        print("❌ No messages parsed.")
        return df

    # ── Parse datetime ─────────────────────────────────────────────────────
    df['datetime'] = pd.to_datetime(
        df['date_str'] + ' ' + df['time_str'],
        format='%d/%m/%Y %I:%M %p',
        errors='coerce'
    )
    # Try alternative format if some failed
    failed = df['datetime'].isna()
    if failed.sum() > 0:
        df.loc[failed, 'datetime'] = pd.to_datetime(
            df.loc[failed, 'date_str'] + ' ' + df.loc[failed, 'time_str'],
            dayfirst=True, errors='coerce'
        )

    df = df.dropna(subset=['datetime'])
    df = df[df['body'].str.len() >= 3]
    df = df[~df['author'].isin(['', 'nan', 'None'])]

    # ── Remove remaining system messages ──────────────────────────────────
    mask = df['body'].str.lower().apply(
        lambda x: not is_system_body(x)
    )
    df = df[mask]

    # Distinct local pseudonyms avoid merging numbers with equal suffixes.
    # These keys are not anonymisation or permission to publish individual data.
    phone_keys = {}
    def normalize_author(name):
        name = name.strip()
        if re.fullmatch(r"\+?[\d\s().-]+", name) and any(c.isdigit() for c in name):
            digits = re.sub(r"\D", "", name)
            if digits not in phone_keys:
                phone_keys[digits] = f"phone_user_{len(phone_keys)+1:03d}"
            return phone_keys[digits]
        return name
    df['author'] = df['author'].apply(normalize_author)

    # ── Sort and detect replies ────────────────────────────────────────────
    df = df.drop_duplicates(['author', 'datetime', 'body'])
    df = df.sort_values('datetime', kind='stable').reset_index(drop=True)
    df['prev_author'] = df['author'].shift(1)
    df['prev_time']   = df['datetime'].shift(1)
    df['gap_mins']    = (df['datetime'] - df['prev_time']).dt.total_seconds() / 60

    df['is_reply'] = (
        (df['gap_mins'] <= 10) &
        (df['prev_author'] != df['author'])
    ).astype(int)

    df['parent_id'] = df.apply(
        lambda r: f"t1_{r.name-1}" if r['is_reply'] else f"t3_{r.name}",
        axis=1
    )

    # ── Final columns ──────────────────────────────────────────────────────
    df['subreddit'] = 'tsec_coders'
    df['score']     = 1
    df = df[['author', 'datetime', 'body', 'parent_id',
             'is_reply', 'subreddit', 'score', 'source']]

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n✅ Clean messages:  {len(df):,}")
    print(f"👥 Unique users:    {df['author'].nunique():,}")
    print(f"📅 Date range:      {df['datetime'].min().date()} → {df['datetime'].max().date()}")
    print(f"↩️  Reply rate:      {df['is_reply'].mean()*100:.1f}%")
    print(f"\nTop 15 most active users:")
    print(df['author'].value_counts().head(15).to_string())

    return df


def main():
    df = parse_nankani(INPUT_FILE)
    if len(df) > 0:
        df.to_json(OUTPUT_FILE, orient="records", lines=True)
        print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
