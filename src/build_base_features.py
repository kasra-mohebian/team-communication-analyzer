"""Build v1 prerequisites from locally supplied, authorised clean corpora."""
from pathlib import Path
import pandas as pd
import config
from features import extract_features, load_clean


def main():
    parts = []
    for dataset, path in config.CLEAN_FILES.items():
        if not Path(path).is_file():
            continue
        frame = extract_features(load_clean(path), label=dataset)
        frame["dataset"] = dataset
        frame["archetype"] = "unlabelled"
        parts.append(frame)
    if not parts:
        raise SystemExit("No authorised local clean corpora found; try src/demo.py instead.")
    pd.concat(parts, ignore_index=True).to_json(config.TRAIN_FEATURES, orient="records", lines=True)
    if Path(config.WHATSAPP_CLEAN).is_file():
        extract_features(load_clean(config.WHATSAPP_CLEAN), label="team").to_json(
            config.WA_FEATURES, orient="records", lines=True)
    print("Base feature matrices written. Labels are a separate, authorised step.")


if __name__ == "__main__":
    main()
