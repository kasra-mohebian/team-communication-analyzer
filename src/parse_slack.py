"""Parse locally supplied Slack XML; inferred chains are scoped to channels.

No inputs are read on import. Missing/zero conversation ids do not establish
reply chains. The output remains private even when its source used aliases.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import config


def parse_slack(data_root, limit=60000):
    files = sorted(Path(data_root).rglob('*.xml'))
    if not files:
        raise ValueError('No Slack XML files found; see docs/DATA.md for local input layout.')
    rows = []
    for path in files:
        relative = path.relative_to(data_root)
        channel = relative.parts[-3] if len(relative.parts) >= 3 else str(relative.parent)
        for msg in ET.parse(path).getroot().findall('message'):
            author = (msg.findtext('user') or '').strip()
            timestamp = (msg.findtext('ts') or '').strip()
            body = (msg.findtext('text') or '').strip()
            if author and timestamp and len(body) >= 3 and author not in {'nan', 'None'}:
                rows.append({'author': author, 'datetime_str': timestamp, 'body': body,
                             'conversation_id': msg.get('conversation_id', '0'),
                             'channel': channel})
    if not rows:
        raise ValueError('Slack XML contains no usable messages.')
    frame = pd.DataFrame(rows)
    # Slack exports may use epoch seconds or ISO date strings, even in one file.
    numeric = pd.to_numeric(frame['datetime_str'], errors='coerce')
    frame['datetime'] = pd.to_datetime(frame['datetime_str'].where(numeric.isna()),
                                       format='mixed', utc=True, errors='coerce')
    mask = numeric.notna()
    frame.loc[mask, 'datetime'] = pd.to_datetime(numeric[mask], unit='s', utc=True, errors='coerce')
    frame = frame.dropna(subset=['datetime']).drop_duplicates(
        ['channel', 'conversation_id', 'author', 'datetime', 'body'])
    frame = frame.sort_values('datetime', kind='stable')
    if len(frame) > limit:
        active = frame['author'].value_counts()
        frame = frame[frame['author'].isin(active[active >= 3].index)].head(limit)
    frame = frame.reset_index(drop=True)
    previous = frame.groupby(['channel', 'conversation_id'], sort=False)['author'].shift(1)
    valid_chain = ~frame['conversation_id'].isin(['', '0', 'None', 'nan'])
    frame['is_reply'] = (valid_chain & previous.notna() & (previous != frame['author'])).astype(int)
    # Chain membership is inferred, not an original platform reply identifier.
    frame['parent_id'] = None
    return frame


def main():
    try:
        frame = parse_slack(Path(config.SLACK_RAW_DIR) / 'data')
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    frame.to_json(config.SLACK_CLEAN, orient='records', lines=True)
    print(f'Parsed {len(frame)} messages in {frame.channel.nunique()} channels; saved local clean stream.')


if __name__ == '__main__':
    main()
