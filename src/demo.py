"""Offline feature demo using invented conversations, with no role-model claims."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re

# Force lexical fallbacks even when the full research environment is installed.
os.environ["SOCIALCOMPASS_OFFLINE"] = "1"

import pandas as pd
from make_synthetic import build_stream
from features import extract_features
from features_v2 import merge_v2, build_interaction_graph


def fictional_rows():
    rows = build_stream()
    names = sorted({r["author"] for r in rows}, key=len, reverse=True)
    aliases = {name: f"Member {i+1}" for i, name in enumerate(sorted(names))}
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b")
    return [{**r, "author": aliases[r["author"]],
             "body": pattern.sub(lambda m: aliases[m.group()], r["body"])} for r in rows]


def run_demo(out):
    rows = fictional_rows()
    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["datetime"], unit="ms")
    parts = []
    graph_edges = []
    # Keep the independent invented teams separate in both feature implementations.
    for team, group in frame.groupby("team", sort=True):
        base = extract_features(group.copy(), label=team, verbose=False)
        part = merge_v2(base, group.copy())
        part["team"] = team
        parts.append(part)
        graph = build_interaction_graph(group)
        graph_edges.extend({"team": team, "from": a, "to": b, "weight": d["weight"]}
                           for a, b, d in graph.edges(data=True))
    features = pd.concat(parts, ignore_index=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic_messages.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    features.to_json(out / "features.json", orient="records", indent=2)
    (out / "edges.json").write_text(json.dumps(graph_edges, indent=2), encoding="utf-8")
    summary = {"synthetic": True, "messages": len(rows), "members": len(features),
               "teams": int(frame["team"].nunique()), "network_edges": len(graph_edges),
               "role_model_used": False, "external_api_calls": 0}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/demo")
    args = parser.parse_args()
    print(json.dumps(run_demo(args.out), indent=2))
    print("Invented conversations only. Features are not predictions about real people.")


if __name__ == "__main__":
    main()
