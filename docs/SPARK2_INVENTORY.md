# behavior-xr1 implementation inventory (Spark2)

Basis: branch `spark2-runtime-integration`, commit `b8927b8b531e964c633739c04963ea2722434c73`; source audit only before edits.

| Milestone | Status | Evidence |
|---|---|---|
| BEHAVIOR ↔ XR-1 interface | PARTIAL | contracts/state/target builder exist; runtime action adapter and observation adapter absent |
| baseline forward path | NOT IMPLEMENTED | no `models/baseline.py`, collate integration, or XR-1 runtime composition |
| 100 task ID plumbing | PARTIAL | committed challenge catalog is documented; no `data/tasks.py` or generated config |
| hybrid language-task conditioning | NOT IMPLEMENTED | no task-conditioning module |
| 31-skill auxiliary head | NOT IMPLEMENTED | no skill head/loss; annotation granularity unconfirmed |
| correlated flow matching | NOT IMPLEMENTED | no noise sampler; upstream seam only documented |
| training config | NOT IMPLEMENTED | no model/data/train config tree |
| checkpoint save/load | PARTIAL | initialization policy tests exist; manifest/checkpoint API absent |
| model smoke test | NOT IMPLEMENTED | no model load/forward script; weights intentionally unavailable |
| OmniGibson evaluation | PARTIAL | scripts for metadata/IK are present; policy server/wrapper absent |

## Baseline

- `compileall`: PASS
- system Python unittest/pytest: BLOCKED by missing numpy/PYTHONPATH and pytest
- existing validation venv + `PYTHONPATH=src`: unittest **18 passed**, pytest **18 passed**
- no code changes or commits have been made beyond branch creation.

## Runtime facts available from reference validation

Isaac Sim 5.1, OmniGibson 3.9.2, R1Pro official 23D action path, 61D proprioception, RGB/depth wrapper, official websocket evaluator and JSON smoke were validated in the separate reference checkout. The reference checkout remains untouched.

## Current blockers

Dataset metadata/rows, full XR-1 weights, and real-model inference were not available and were not downloaded. IK and action semantic reconstruction remain data-dependent. The repository's WSL/A100/v3.9.1 assumptions require documentation update before implementation claims; no unsupported mapping is being added.
