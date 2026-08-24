# Implementation plan

No research feature below is implemented in the current milestone. Shape
notation: `B` batch, `T` time, `S` language/vision token count, `H_vlm` Qwen
hidden width. BEHAVIOR dimensions below refer only to the bundled R1Pro config.

| # | Milestone | Planned changes / new files | Inputs → outputs | Test | A100? |
|---:|---|---|---|---|---|
| 1 | BEHAVIOR ↔ XR-1 interface | Current: audited immutable layouts in `data/schema.py`, fail-safe adapter interfaces in `adapters/contracts.py`. Future: `data/observation.py`, state/action packers only after semantic decision. Keep upstream untouched. | camera dict + `[B,61]` + default `[B,T,23]` (or unverified IK `[B,T,21]`) ↔ Qwen messages + `[B,1,60]` + `[B,30,60]` | Current CPU layout/non-overlap/NotImplemented tests; later frame/scale and simulator-controller contract smoke test | No for current tests; yes for controller verification |
| 2 | Baseline forward path | New `models/baseline.py`, `data/collate.py`; thin composition around vendor-compatible XR-1 APIs | adapted Qwen tensors, state `[B,1,60]` → action chunk `[B,30,60]` → R1Pro `[B,K,23]` | mock VLM/DiT CPU shape test; never initialize Qwen locally | Yes for real model |
| 3 | 100 task ID plumbing | New `data/tasks.py`, generated small `configs/tasks.json`; update collate/schema | canonical string ID / dataset task index → stable `[B]` integer ID and language text | validate uniqueness/cardinality=100 and mapping round-trip against pinned `task_data.json` | No; dataset mapping confirmation needs data host |
| 4 | Hybrid language-task conditioning | New `models/task_conditioning.py`; minimal hook in future XR-1 compatibility wrapper; config fields | VLM language/vision representation `[B,S,H_vlm]` + task IDs `[B]` → fused conditioning/cache-compatible representation; DiT width 1024 where projected | tiny configurable-dimension module test; checkpoint key compatibility test | Yes for XR-1 integration |
| 5 | 31-skill auxiliary head | New `models/skill_head.py`, `training/skill_loss.py`; dataset label alignment | selected Qwen representation `[B,S,H_vlm]` and labels/mask `[B,T]` or `[B]` (granularity **TBD**) → logits `[...,31]`, auxiliary CE | synthetic masked-loss/gradient test; verify no logits/predictions enter DiT arguments | Yes for real representations |
| 6 | Correlated flow matching | New `training/noise.py`; configurable noise sampler injected at XR-1 Gaussian seam corresponding to upstream `XR1.py:399`; loss wrapper if required | action `[B,30,60]`, mask → correlated noise `[B,30,60]`; preserve velocity target shape | covariance/statistical CPU test, iid regression, deterministic seed; paper/code parity test after audit | No for unit tests; yes for model experiment |
| 7 | Training config | New `configs/model/*.yaml`, `configs/data/*.yaml`, `configs/train/*.yaml`; config parser | paths, freeze policy, loss weights, adapter metadata → validated immutable config | parse-only tests, reject laptop/heavy execution profile | No |
| 8 | Checkpoint save/load | New `training/checkpoint.py`; compatibility manifest including upstream commits, task map, pack schemas, normalization | XR-1 base state + adapter/task/skill parameters → resumable A100 checkpoint | tiny-module strict/non-strict tests; missing/mismatched schema rejection | No for unit; yes for full checkpoint |
| 9 | A100 smoke test | New `scripts/a100_smoke.py`, runbook | one metadata/sample batch → finite forward/loss/backward and action chunk | single batch, no full training; memory/timing report; checkpoint reload equivalence | Yes |
| 10 | OmniGibson evaluation | New `evaluation/policy_server.py`, `evaluation/behavior_wrapper.py`, runbook | flattened RGB/depth/proprio websocket observation → one exact `robot.action_dim` action | one public instance first, then official RGBDFullResWrapper protocol; validate no privileged inputs | Yes + OmniGibson/assets |

## Immediate next implementation slice

The current static decision favours arm IK (`pose_delta_ori`) for XR-1 native
EE-action preservation, but only after A100 checks confirm R1Pro IK action
dimension/link frames and metadata checks establish a valid demo-target
reconstruction path. Start next with metadata-only inspection, then an A100
controller smoke test. Do not choose silent truncation, padding, or arbitrary
slicing to resolve semantic mismatches.
