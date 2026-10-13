# behavior-xr1

Research scaffold for adapting Xiaomi Robotics XR-1 to the 2026 BEHAVIOR
Challenge. The upstream repositories are read-only references; all adaptation
code belongs here.

This checkout is intentionally lightweight. Do not install or launch
OmniGibson / Isaac Sim, download BEHAVIOR datasets or assets, download XR-1 or
Qwen checkpoints, install CUDA / FlashAttention, or train models on this
machine. Training, inference, and simulator evaluation belong on the later
Ubuntu A100 host.

The current milestone contains architecture analysis and interface contracts
only. See `docs/ARCHITECTURE.md` and `docs/IMPLEMENTATION_PLAN.md`.

## Lightweight checks

```bash
python3 -m compileall -q src tests
python3 -m unittest discover -s tests -v
```

## 2026 BEHAVIOR handoff status

This checkout contains the verified interface work for adapting the original Xiaomi XR-1 5B model to the 2026 BEHAVIOR R1Pro evaluation path. The current handoff selection is task IDs `0000`–`0009` (Task 0 is `turning_on_radio`); the training team owns confirmation of that split against the official 2026 manifest. The complete production task manifest and normalization statistics are not checked into this repository.

Verified contracts:

- **State:** R1Pro 61D observation → XR-1 dense 60D state (`src/behavior_xr1/data/r1pro_2026_state.py`).
- **Model:** original XR-1 5B checkpoint; verified output shape `(1, 30, 60)`.
- **Evaluation action:** XR-1 60D → configured R1Pro controller 21D (`src/behavior_xr1/adapters/action_bridge.py`).
- **Gripper:** positive command opens; negative command closes, established by the 90-step Task 0 calibration artifact.
- **Normalization:** final compatible 2026 normalization is **not available**. The checked-in validation statistics are not production statistics.
- **Zero-shot rollout:** not executed; no predicted action has been run through a policy rollout.
- **Evaluation runner:** `src/behavior_xr1/evaluation/run_zero_shot.py` is currently a contract validator only. It intentionally refuses model-backed execution until compatible normalization and chunk-execution semantics are supplied.
- **Task conditioning:** production XR-1 task-conditioning integration is outside this focused handoff commit; its checkpoint and train/eval parity contract remain a training-team gate.

The remaining production gates are a training-team-owned 2026 normalization artifact (state q01/q99 `(1,60)`, action mean/std `(30,60)` with provenance), a genuine model-backed evaluation runner with verified observation refresh/replanning semantics, and the final train/eval state-contract sign-off. Do not use 2025 N2 statistics as a substitute.

For lightweight checks:

```bash
python3 -m compileall -q src tests
python3 -m unittest discover -s tests -v
```

Large datasets, checkpoints, virtual environments, generated logs, simulator caches, and N2 intermediate artifacts are intentionally excluded by `.gitignore` and by the focused commit for this handoff. Runtime paths in documentation are illustrative; configure local dataset/checkpoint roots explicitly rather than relying on absolute machine paths.
