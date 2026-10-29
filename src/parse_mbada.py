"""
parse_mbada.py — Mbada 2023 WhatsApp export (.docx) → data/clean/mbada_clean.json

Source
------
Mbada et al. (2023), a professional WhatsApp group of Nigerian physiotherapists,
published as a Word document. The authors already pseudonymized every
participant to "Anon N" / "Contact N" before publication, so this parser never
sees a real identity.

Why this dataset is in the project
----------------------------------
Slack and the Nankani CODERS chat are technical Q&A: they contain almost no
Butterfly (information bridge) or Capybara (warm supporter) behaviour. Mbada is
a professional-but-social group — congratulations, condolences, link sharing —
and supplies exactly the two rare classes the technical corpora starve.

Document format
---------------
Each message starts a new paragraph:

    8:28 PM - ANON 1: *ACAPN keeps making headlines*
    The above *press release* is a must read!          <- continuation paragraph
    8:32 PM – Anon 2: 💪💯                              <- note the EN DASH
    1/8/21, 5:12 AM - Anon 61: 🍓 *Morning Tea*        <- some lines carry a date

Handled quirks:
  * separator is "-", "–" or "—" depending on the line
  * author case is inconsistent ("ANON 1" vs "Anon 3") → lower-cased
  * a message body may continue over any number of following paragraphs
  * `&lt;` / `&amp;` etc. must be unescaped — they are raw XML entities
  * only real participants ("anon N") are kept; "Contact N" is a non-member
    whose messages were forwarded/quoted into the group

Timestamps — IMPORTANT LIMITATION
---------------------------------
The document gives a time of day but no reliable, continuous date, so real
inter-message gaps cannot be recovered. We therefore emit a synthetic 60-second
ladder (0, 60000, 120000, … ms), which is what the original extraction used and
what the committed feature matrix was built from.

Consequence: timing features are artifacts of the constructed ordering, not
measurements of real reply delays. They may vary by participant, so no genuine
timing claim should be made for this corpus. See docs/DATA.md.

Usage
-----
    python src/parse_mbada.py
    python src/parse_mbada.py --verify   # compare against the existing clean file
"""
import html
import json
import re
import sys
import zipfile
from pathlib import Path

import config
from message_cleaning import is_system_body

MBADA_DOCX_GLOB = "Mbada*.docx"

# "8:28 PM - ANON 1: body"   /   "1/8/21, 5:12 AM – Anon 61: body"
HEADER = re.compile(
    r"^(?:\d{1,2}/\d{1,2}/\d{2,4},\s*)?"      # optional leading date
    r"(\d{1,2}:\d{2})\s*(AM|PM|am|pm)?\s*"    # time of day
    r"[-–—]\s*"                               # hyphen / en dash / em dash
    r"([^:]{1,40}?):\s*"                      # author
    r"(.*)$"                                  # body
)

# real participants only; "Contact N" is a quoted outsider, not a group member
PARTICIPANT = re.compile(r"^anon\s*\d+$")

SYSTEM_PHRASES = [
    "<media omitted>", "this message was deleted", "you deleted this message",
    "messages and calls are end-to-end encrypted", "image omitted",
    "video omitted", "audio omitted", "document omitted", "sticker omitted",
    "(file attached)", "null",
]

STEP_MS = 60_000          # synthetic 60-second ladder (see module docstring)


def docx_paragraphs(path):
    """Yield the visible text of every paragraph in a .docx, in order.

    python-docx would do this too, but a .docx is just a zip of XML and the two
    regexes below keep the project's dependency list smaller.
    """
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S))
        text = html.unescape(text).strip()
        if text:
            yield text


def parse_mbada(path):
    messages = []
    for line in docx_paragraphs(path):
        m = HEADER.match(line)
        if m:
            author = re.sub(r"\s+", " ", m.group(3).strip().lower())
            messages.append({"author": author, "body": m.group(4).strip()})
        elif messages:
            # continuation paragraph of the message currently being built
            messages[-1]["body"] = (messages[-1]["body"] + " " + line).strip()

    rows = []
    for m in messages:
        if not PARTICIPANT.match(m["author"]):
            continue                                   # non-member / quoted contact
        body = m["body"]
        if not body:
            continue
        if is_system_body(body):
            continue                                   # media / deleted / system
        rows.append({
            "author": m["author"],
            "datetime": len(rows) * STEP_MS,           # synthetic — see docstring
            "body": body,
            "is_reply": False,                         # no thread info in the export
            "parent_id": None,
        })
    return rows


def find_docx():
    hits = sorted(Path(config.RAW).glob(MBADA_DOCX_GLOB))
    if not hits:
        raise SystemExit(
            f"No {MBADA_DOCX_GLOB} in {config.RAW}.\n"
            "The raw corpora are not committed — see docs/DATA.md for how to "
            "obtain them.")
    return hits[0]


def main():
    src = find_docx()
    rows = parse_mbada(src)
    authors = {r["author"] for r in rows}

    print(f"✅ Parsed {len(rows)} messages from {src.name}")
    print(f"👥 Unique participants: {len(authors)}")
    print("⚠  Timestamps are synthetic (60 s ladder) — timing features are not "
          "real behavioural evidence; see docs/DATA.md.")

    if "--verify" in sys.argv:
        out = Path(config.MBADA_CLEAN)
        if not out.exists():
            print(f"   (nothing to verify against — {out} does not exist)")
        else:
            ref = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
            import collections
            a = collections.Counter((r["author"], r["body"]) for r in rows)
            b = collections.Counter((r["author"], r["body"]) for r in ref)
            same = sum((a & b).values())
            print(f"   verify: {same}/{len(ref)} messages identical to the "
                  f"existing clean file ({100 * same / len(ref):.2f}%)")
        return

    with open(config.MBADA_CLEAN, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✅ Saved to {config.MBADA_CLEAN}")


if __name__ == "__main__":
    main()
