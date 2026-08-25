#!/usr/bin/env python3
"""Read a bounded set of parquet rows; video files are never opened."""

from __future__ import annotations

import argparse
from pathlib import Path


def _find_episode(root: Path, episode: int) -> Path:
    patterns = (f"episode_{episode:06d}.parquet", f"episode_{episode}.parquet", f"*{episode:06d}*.parquet")
    matches = []
    for pattern in patterns:
        matches.extend(root.glob(f"data/**/{pattern}"))
        matches.extend(root.glob(pattern))
    unique = list(dict.fromkeys(sorted(matches)))
    if len(unique) != 1:
        raise FileNotFoundError(f"expected one parquet for episode {episode}; found {unique}")
    return unique[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", required=True, help="printed label; does not select or load video")
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--num-rows", type=int, default=5)
    args = parser.parse_args()
    if args.num_rows < 1:
        parser.error("--num-rows must be positive")
    try:
        import pyarrow.parquet as pq  # Delayed so --help needs no heavy/data dependency.
    except ImportError as error:
        parser.error(f"pyarrow is required on the data host: {error}")
    path = _find_episode(args.dataset_root.expanduser().resolve(), args.episode)
    table = pq.read_table(path).slice(0, args.num_rows)
    print(f"task: {args.task}\nepisode: {args.episode}\nparquet: {path}\nrows: {table.num_rows}")
    print("schema:")
    print(table.schema)
    rows = table.to_pylist()
    for index, row in enumerate(rows):
        action = row.get("action")
        state = row.get("observation.state", row.get("observation/state"))
        left = state[24:26] if state is not None and len(state) >= 26 else None
        right = state[49:51] if state is not None and len(state) >= 51 else None
        metadata = {key: row.get(key) for key in ("task", "task_index", "frame_index", "index", "timestamp", "episode_index") if key in row}
        print(f"\nrow {index}: metadata={metadata}")
        print(f"action shape={getattr(action, 'shape', (len(action),) if action is not None else None)} value={action}")
        print(f"state shape={getattr(state, 'shape', (len(state),) if state is not None else None)} value={state}")
        print(f"gripper qpos pairs: left={left}, right={right}")


if __name__ == "__main__":
    main()
