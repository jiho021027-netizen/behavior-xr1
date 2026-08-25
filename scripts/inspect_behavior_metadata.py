#!/usr/bin/env python3
"""Print BEHAVIOR/LeRobot metadata only; it never opens videos or episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _show_json(label: str, path: Path) -> None:
    print(f"\n== {label}: {path} ==")
    try:
        print(json.dumps(json.loads(path.read_text()), indent=2, sort_keys=True))
    except (OSError, json.JSONDecodeError) as error:
        print(f"unreadable: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    print(f"dataset_root: {root}")
    for name in ("info.json", "tasks.json", "stats.json"):
        matches = sorted(root.glob(f"meta/**/{name}")) + sorted(root.glob(name))
        if not matches:
            print(f"\n== {name} ==\nnot found")
        for path in dict.fromkeys(matches):
            _show_json(name, path)


if __name__ == "__main__":
    main()
