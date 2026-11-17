"""Check agent output contracts and document-group separation in all three splits."""

import argparse
import json
from collections import Counter
from pathlib import Path

from training_lab.engine.contracts import LAB, load_splits, split_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=LAB / "data")
    args = parser.parse_args()
    rows = load_splits(args.dataset)
    print(
        json.dumps(
            {
                "tasks": {
                    split: Counter(item.task for item in records) for split, records in rows.items()
                },
                "sha256": split_hashes(args.dataset),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
