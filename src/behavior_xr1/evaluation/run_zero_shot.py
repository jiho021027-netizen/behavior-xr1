"""Dedicated 2026 BEHAVIOR zero-shot evaluation entrypoint.

The runner deliberately refuses policy execution without a compatible 60D
normalization artifact and an explicit checkpoint.  ``--dry-run`` validates
observation/state/action contracts without stepping the simulator.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ACTIVE = np.array(list(range(0, 7)) + list(range(8, 15)) + list(range(20, 24)), dtype=np.int64)

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--instance-id", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-env-steps", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--one-policy-step", action="store_true")
    p.add_argument("--rollout", action="store_true")
    return p

def load_normalization(path: Path) -> dict:
    d = json.loads(path.read_text())
    q01 = np.asarray(d.get("state_q01", []), dtype=np.float32)
    q99 = np.asarray(d.get("state_q99", []), dtype=np.float32)
    mean = np.asarray(d.get("action_mean", []), dtype=np.float32)
    std = np.asarray(d.get("action_std", []), dtype=np.float32)
    if q01.shape != (1, 60) or q99.shape != (1, 60):
        raise ValueError("normalization state quantiles must be (1,60)")
    if mean.shape != (30, 60) or std.shape != (30, 60):
        raise ValueError("normalization action statistics must be (30,60)")
    if not all(np.isfinite(x).all() for x in (q01, q99, mean, std)):
        raise ValueError("normalization contains non-finite values")
    return d

def main(argv=None) -> int:
    a = parser().parse_args(argv); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    result = {"task_id": a.task_id, "instance_id": a.instance_id, "checkpoint": str(Path(a.checkpoint)),
              "normalization": str(Path(a.normalization)), "active_action_indices": ACTIVE.tolist(),
              "action_horizon": 30, "env_action_dim": 21, "dry_run": a.dry_run}
    if not Path(a.checkpoint).exists(): raise FileNotFoundError(a.checkpoint)
    stats = load_normalization(Path(a.normalization)); result["normalization_compatible"] = True
    if a.dry_run:
        result.update({"status": "DRY_RUN_CONTRACT_PASS", "policy_execution": False,
                       "gripper_actions_enabled": False})
    else:
        raise RuntimeError("real policy execution is gated until verified 2026 normalization and gripper semantics are available")
    (out / "zero_shot_dry_run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
