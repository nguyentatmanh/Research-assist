"""Prepare raw train.json, validation.json, test.json into evidence-level experiments."""

import argparse
import json
from pathlib import Path

from research_assist.experiments.data import prepare_qasper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/datasets/qasper"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/qasper"))
    args = parser.parse_args()
    manifest = prepare_qasper(args.input, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "statistics": manifest["statistics"],
                "cross_split_duplicate_texts": manifest["cross_split_duplicate_texts"],
            },
            indent=2,
        )
    )
