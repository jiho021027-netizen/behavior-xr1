# Architecture audit

Audit basis: XR-1 commit `556cca33963a2b36d835a40374c3b4c8eef68401`
and BEHAVIOR-1K tag `v3.9.1` / commit
`26f2c7ef7b9cf96bd0414f81e1e751e493762779`. Paths below are relative to the
corresponding upstream root. “Inference” labels our interpretation; all other
claims are directly visible in the cited source.

## XR-1

| Question | Confirmed code location and symbol |
|---|---|
| VLA / pretrained VLM owner | `xr1/mibot/models/VLA/XR1.py:197`, class `xr1`; `_build_model()` at line 219 loads `Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")` and constructs `Qwen3VLForConditionalGeneration._from_config(...)`. This config lookup may access Hugging Face and must not run on this laptop. |
| Qwen3-VL integration | Modified implementation in `xr1/mibot/models/VLM/qwen3vl.py`: `Qwen3VLModel` and `Qwen3VLForConditionalGeneration`. Its `Qwen3VLModel.forward()` replaces `<state>` and action/score token embeddings. Token constants and processor integration are in `xr1/mibot/data/collate/custom_collate.py`, class `CustomCollate`. |
| DiT | `xr1/mibot/models/VLA/XR1.py:165`, class `DiT`, made of `DecoderLayer`; `xr1.dit_forward()` at line 298 combines sink, projected state, and noisy actions and attends to VLM KV caches. |
| State projectors | `xr1._build_model()`: `state_projector_choice` maps 60 to VLM hidden size for the VLM action-choice branch; `state_projector` maps 60 to DiT width 1024. |
| Action projectors | `action_projector_choice` maps VLM hidden states to `60 * n_choices`; DiT `action_projector` maps 60 to 1024 and `action_output_layer` maps 1024 back to 60. |
| Shapes | `xr1.__init__()` fixes `state_shape=(1,60)`, `action_shape=(30,60)`. `docs/data_format.md` defines occupied action slots: left EE delta 0:7, right EE delta 8:15, waist 16, base velocity 17:20; remaining slots reserved. State slot semantics are produced by `compose_state()` in `xr1/mibot/utils/io.py` and normalized with q01/q99. |
| Flow matching | `xr1.forward()`: sample Gaussian `noise`; interpolate `noisy_action=(1-t)*noise+t*action`; velocity target is `action-noise`; `dit_forward()` predicts it. `compute_flow_loss()` applies weighted masked MSE plus an rFFT frequency loss. Current Gaussian insertion point is line 399 (`torch.randn_like(action)`), the natural future correlated-noise seam. |
| Language tokens | `JsonDataset._prompt()` selects `instruction.general`; `_messages()` interleaves image and text content. `JsonDataset.__getitem__()` appends `Robot state: <state>` and action/score special tokens. `CustomCollate.__call__()` invokes Qwen chat templating and constructs packed sequence/position/condition segments. |
| Checkpoint loading | Training: `xr1/mibot/models/runner/base_runner.py`, `BaseRunner.configure_model()`, loads `params.pretrained` with `torch.load(...)["module"]`, `strict=False` followed by an explicit mismatch error. Deployment: `xr1/mibot/server/deploy.py::load_model()` loads `last.ckpt/checkpoint/mp_rank_00_model_states.pt`, strips `model.`, and uses `strict=True`. `xr1/tools/weight_convert.py` converts a Hugging Face checkpoint. |
| Post-training entrypoint | `xr1/tools/train.py::main()` (Hydra) builds the data/model through `prepare()` and calls Lightning `Trainer.fit`; shell entry is `xr1/scripts/train.sh`; configuration starts at `xr1/configs/config.yaml` and `xr1/configs/model/posttrain.yaml`. |
| Dataset / dataloader | `xr1/mibot/data/datasets/json_dataset.py::JsonDataset`; `xr1/mibot/data/datamodule/base_datamodule.py::BaseDataModule.train_dataloader`; collator is `CustomCollate`. Input schema is documented in `xr1/docs/data_format.md`. |

Important preservation fact: `_build_model()` freezes only the VLM input
embedding (`get_input_embeddings().requires_grad_(False)`); it does not freeze
the complete pretrained VLM. A resource-efficient fine-tuning policy therefore
needs an explicit parameter-freezing configuration rather than assuming the
released post-training code already preserves all VLM weights.

## BEHAVIOR 2026

Official repository documentation requires tag `v3.9.1` for challenge
evaluation/replay. The challenge overview states 100 tasks and 20,000 demos;
`docs/challenge/dataset.md` states LeRobot v3, 3.27 TB (not downloaded), and raw
HDF5 1.44 TB (not downloaded).

| Question | Confirmed source / result |
|---|---|
| 100 task definitions / mapping | `docs/challenge/task_data.json` contains exactly 100 unique `id`/display-`name` entries under `tasks`; only the first 50 entries also contain `instruction`. This is the committed challenge catalog/demo-gallery mapping. `docs/gen_2026_task_data.py` documents its source generation. Executable symbolic definitions live under `bddl3/bddl/activity_definitions/<task_id>/problem0.bddl` (for example `turning_on_radio/problem0.bddl`). Dataset-side natural language for all 100 is documented as `meta/tasks.jsonl` but is not in this checkout. |
| Demonstration format | `docs/challenge/dataset.md`: LeRobot v3 layout `annotations/`, `data/`, `meta/`, `videos/`; 20,000 demos. Low-dimensional data include proprioception/actions; videos include RGB/depth. Exact parquet feature schema is **not confirmed locally** because dataset metadata was not downloaded. |
| Observations | Challenge evaluation allows RGB + depth + proprioception only. `RGBDFullResWrapper` in `OmniGibson/omnigibson/eval/wrappers/rgbd_full_res_wrapper.py` enables `rgb` and `depth_linear`; head is 720x720 and wrists 480x480. Dataset docs describe RGB `(H,W,4)` uint8 and depth `(H,W)` float32 in [0,10] m. |
| Default proprioception | `OmniGibson/omnigibson/eval/r1pro.yaml::proprio_obs` lists 15 components. `OmniGibson/omnigibson/eval/utils/eval_utils.py::PROPRIOCEPTION_INDICES["R1Pro"]` maps them over indices 0:61, hence 61 dimensions. The first three are robot-local base velocity per `docs/challenge/updates.md`. |
| Default action | `r1pro.yaml::controller_config` selects base velocity (3), trunk joint position (4), two arm joint positions (7+7), and two scalar gripper commands (1+1); camera is null. `eval_utils.py::ACTION_QPOS_INDICES["R1Pro"]` confirms slices 0:23. `action_normalize: false`. Runtime truth remains `robot.action_dim`; custom robots/controllers are allowed, so 23 is specifically the bundled R1Pro config, not a universal challenge constant. |
| Robot embodiment | Default is R1Pro (`r1pro.yaml`), but `docs/challenge/evaluation.md` explicitly permits any OmniGibson-supported custom robot configuration. |
| Evaluation entrypoint | `OmniGibson/omnigibson/eval/eval.py::main()`, invoked with `python -m omnigibson.eval.eval`; `Evaluator` sets the policy action dimension from `self.robot.action_dim`. Websocket observations are recursively flattened by `eval_utils.flatten_obs_dict()`. |
| Task ID / name | CLI `--task-name` accepts the string activity ID (example `turning_on_radio`). The 100 committed `id`/`name` pairs are in `docs/challenge/task_data.json`. Dataset integer `task_index` mapping and `meta/tasks.jsonl` schema are **not confirmed locally**. |
| 31 skills | Names and aggregate statistics are committed in `docs/challenge/dataset.md`: attach, chop, close door, close drawer, close lid, hand over, hang, hold, ignite, insert, move to, open door, open drawer, open lid, pick up from, place in, place in next to, place on, place on next to, place under, pour, press, push to, release, spray, sweep surface, tip over, turn off switch, turn on switch, turn to, wipe hard. Exact per-frame/per-segment annotation files, keys, class IDs, boundaries, and alignment timestamps are **not confirmed locally**; they require inspecting only the small Hugging Face metadata/annotations on the data host, without downloading videos/parquet shards. |

## XR-1 ↔ BEHAVIOR interface gap

```text
BEHAVIOR RGB/depth + flattened camera keys
        -> BehaviorInputAdapter (select/order views; depth encoding policy)
        -> Qwen3-VL messages/pixel inputs

BEHAVIOR R1Pro proprio [B, 61]
        -> BehaviorStatePacker (explicit semantics/normalization/mask)
        -> XR-1 state [B, 1, 60]

BEHAVIOR R1Pro action [B, T, 23]
        <- BehaviorActionUnpacker (controller semantics/units/chunk policy)
        <- XR-1 packed action [B, 30, 60]
```

| Interface | BEHAVIOR v3.9.1 default R1Pro | XR-1 released code | Gap / required decision |
|---|---:|---:|---|
| State | `[B,61]` ordered by `PROPRIOCEPTION_INDICES` | `[B,1,60]` | One-dimensional mismatch plus fundamentally different semantics: BEHAVIOR includes velocities, EE poses/quaternions, gripper joints, trunk; XR-1 `compose_state()` uses its own packing. No truncation is justified. |
| Action | per-step `[B,23]` joint/controller command | chunk `[B,30,60]`; only 20 slots occupied in the documented source embodiment | Must map base/trunk/arms/grippers between joint-space R1Pro controls and XR-1 EE-delta layout. This likely needs kinematics or a deliberate alternative XR-1 packing; not implemented. |
| Images | three named cameras, RGBA and metric depth | JSON loader expects prompt-selected PIL image views and Qwen image tokens; no depth-specific path | Define RGBA→RGB, view ordering, depth representation (or RGB-only baseline), resizing, and evaluation key matching. |
| Language/task | all 100 task string IDs are locally available; gallery natural-language instruction exists for only 50, while docs say dataset `meta/tasks.jsonl` covers all 100; runtime task-ID observation key is not confirmed for the 2026 wrapper | language is chat text; no trainable task embedding | Preserve language path; add a separately indexed 100-task embedding only after dataset/runtime task and language mapping are verified. |
| Skills | 31 names confirmed; annotation schema unconfirmed | no skill auxiliary head | Add VLM-representation-only classifier after locating label granularity/alignment; do not feed predicted skill into DiT. |
| Noise | no dataset constraint | iid Gaussian at `XR1.py:399` | Future correlated sampler can replace noise construction while preserving interpolation and loss API; exact 2025 winning method is not yet audited here. |

### Explicit inferences

- A 61→60 slice is not a valid adapter merely because the dimensions differ by
  one; semantics and normalization differ.
- The documented 23→60 action mismatch cannot be solved by zero-padding alone:
  BEHAVIOR uses the configured joint controllers while XR-1's occupied slots are
  EE deltas plus base/waist commands.
- The VLM KV cache consumed by every DiT layer is the clean insertion region for
  language/task conditioning, but the exact fusion design remains a future
  implementation decision.

## Metadata-only A100/data-host TODO

1. Inspect `meta/info.json`, `meta/tasks.jsonl`, one `meta/episodes/*.jsonl`, and
   one annotation record without fetching media or parquet shards where possible.
2. Confirm feature names, dtypes, shapes, sampling rate, task-index ordering,
   language annotations, skill IDs, skill boundary convention, and timestamps.
3. Inspect one small parquet row group only if needed to verify the 61-state and
   23-action arrays and their units; do not fetch the full dataset.
4. Instantiate R1Pro only on the A100 simulator host to assert `robot.action_dim`
   and actual flattened websocket keys against these static findings.
