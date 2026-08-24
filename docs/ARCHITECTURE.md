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

## R1Pro Action-Space Decision

### Default Joint-Space Path

The official evaluation configuration is
`OmniGibson/omnigibson/eval/r1pro.yaml`. It selects a velocity
`HolonomicBaseJointController`, position `JointController`s for trunk and both
arms, and smooth `MultiFingerGripperController`s. It explicitly sets
`action_normalize: false`. The official R1Pro slice table is
`OmniGibson/omnigibson/eval/utils/eval_utils.py::ACTION_QPOS_INDICES`:

| Slice | Controller command | Confirmed semantics |
|---|---|---|
| `0:3` | base | `[vx, vy, wz]`, robot-local; controller is velocity mode. `HolonomicBaseJointController._update_goal()` rotates planar velocity into canonical coordinates internally. YAML input limits are `[-1,1]`; output limits are `[-.75,-.75,-1]` to `[.75,.75,1]`. |
| `3:7` | torso/trunk | Four position `JointController` commands, `use_delta_commands: false`; input/output limits are `null`, so they are direct controller commands when normalization is disabled. |
| `7:14` | left arm | Seven position `JointController` commands, `use_delta_commands: false`; input/output limits are `null`, so direct absolute joint targets. |
| `14:15` | left gripper | One `mode: smooth` MultiFinger command; it is broadcast to both controlled finger joints. The exact converted position range depends on R1Pro asset joint limits. |
| `15:22` | right arm | Same seven direct absolute joint-target commands. |
| `22:23` | right gripper | Same one smooth broadcast gripper command. |

`Robot.action_dim` is the sum of each loaded controller's `command_dim`
(`robots/robot.py::action_dim`); input slices are emitted in
`controller_order` (`Robot.apply_action`). `JointController.command_dim` is
`len(dof_idx)` and smooth gripper `command_dim` is one. The generic robot
definition supplying the literal R1Pro `raw_controller_order` is asset-backed
and not present in this source-only checkout; the above 23D order is instead
confirmed by the official R1Pro evaluation slice table.

If `action_normalize=True`, `Robot._load_controllers()` overwrites every
controller input limit with `"default"`, i.e. `[-1,1]`. `BaseController` clips
and linearly maps input to output only when *both* limits are set. This does
not make the joint controller a delta controller: with the evaluator YAML its
output limit remains `null`, so an arm/trunk command remains a direct absolute
joint target numerically restricted to `[-1,1]`.

### IK / EE-Space Path

`InverseKinematicsController` is source-supported for every manipulation arm:
`Robot._default_arm_ik_controller_configs` builds one for each `arm_names`
entry with mode `pose_delta_ori`; config resolution permits selecting it by
name. This is evidence of framework/R1Pro configuration support, but an actual
R1Pro instantiation remains an A100 simulator-host check.

`controllers/ik_controller.py::IK_MODE_COMMAND_DIMS` gives `pose_delta_ori`
six commands per arm: local-base-frame `[dx,dy,dz,dax,day,daz]`. `_update_goal`
adds translation to the current EEF pose relative to robot base and composes
axis-angle rotation with its current orientation. Generic default arm IK output
limits are translation ±0.2 and axis-angle ±0.5; default input limits are
`[-1,1]`, so linear scaling applies even when the evaluator's
`action_normalize` remains false. The controller applies IK to produce joint
position targets.

Keeping base (3), trunk (4), and smooth grippers (1+1), then replacing both
7D joint arm controllers with 6D IK controllers, yields **21D**:
`3 + 4 + 6 + 1 + 6 + 1`. This total is a source-backed arithmetic consequence,
not a simulator-instantiated `robot.action_dim`; record it as an inference
until A100 validation.

### XR-1 Native Representation

`xr1/mibot/utils/io.py::ACTION_PARTS`, `JsonDataset._arm_action()`, and
`recover_action()` define XR-1's native 30×60 action horizon. Occupied slots
are left EE local translation delta `0:3`, local axis-angle delta `3:6`,
gripper delta `6:7`; right equivalents `8:11`, `11:14`, `14:15`; waist delta
`16:17`; base velocity `17:20`. Slots `7`, `15`, and `20:60` are zero-filled
and excluded by `build_action_mask()`. Arm delta construction rotates world
target displacement by the current EEF rotation transpose; recovery rotates it
back. Therefore XR-1's arm action is EEF-local, whereas OmniGibson IK position
commands are documented relative to the **robot base**. An EEF-to-base frame
transform is required even with IK.

`compose_state()` writes left joint `0:7`, left gripper `7`, right joint
`8:15`, right gripper `15`; it zero-pads `16:60`. State q01/q99 quantile
normalization maps valid dimensions to `[-1,1]`; action uses per-horizon-step
mean/std normalization. Exact physical units are not declared in upstream
XR-1 data-format documentation, so they are not assumed here.

### Recommended Path

**Recommended conditional direction: Strategy B, R1Pro arms configured as
`InverseKinematicsController(mode="pose_delta_ori")`, while retaining default
base/trunk/grippers.** It is closer to XR-1's six-DoF relative EE actions and
eliminates runtime EE-to-joint IK in the policy adapter. It is not a direct
identity: XR-1 is EEF-local while OmniGibson IK deltas are robot-base-relative,
and gripper/waist/controller scaling remain different.

This recommendation is conditional because released 2026 demonstrations have
strong repository evidence for width-23 action records, consistent with the
default joint-controller interface. An IK evaluation policy will need a
training-target reconstruction decision and A100 simulator verification. If
that reconstruction is not validated, retain Strategy A for data/evaluation
compatibility despite its larger semantic gap.

| Property | Strategy A: default joints | Strategy B: arm IK |
|---|---|---|
| Evaluator output | 23D official default | inferred 21D custom robot config |
| Direct XR-1 overlap | base velocity only; gripper/waist only semantically partial | base velocity plus 6D EE delta structure per arm |
| Separate kinematics | XR-1 EEF actions must be converted to joint targets | controller performs IK; adapter still transforms EEF-local → base frame |
| Normalization | controller-specific; arms/trunk direct with config's null limits | IK `[-1,1]` → ±0.2 m/±0.5 rad default limits; others unchanged |
| Demo compatibility | strongest: 23D raw/parquet action tooling | unknown: demos are not confirmed as IK commands |
| Evaluation compatibility | official bundled config | permitted custom config, must submit it |
| Complexity / loss | lower runtime complexity but joint/EE semantic mismatch | lower action-semantic mismatch; reconstruction and frame/scaling work needed |

### Remaining Unknowns

1. The exact R1Pro asset definition's `raw_controller_order`, joint names,
   limits, and EEF link transforms are unavailable without simulator assets.
2. A live R1Pro IK load must confirm each controller's link name and total 21D
   `robot.action_dim`.
3. 2026 LeRobot metadata/parquet must establish all action columns' controller
   provenance and units. `update_lerobot_base_qvel.py` defaults to action width
   23 and compares raw HDF5 action with parquet within a configured tolerance;
   it is strong evidence of 23D action
   storage and base `action[t]` correspondence, but not an episode-level
   controller-config manifest.
4. Deterministic reconstruction of IK targets from demos is unconfirmed. The
   61D proprio state contains EEF positions/quaternions and arm joint values,
   but dataset frame alignment, future-target availability, and controller
   scaling must be inspected from small metadata/data samples on the A100 host.
5. XR-1 source does not declare the coordinate frame/units for its base velocity
   field, nor physical units for its joint/gripper/waist state fields.

## Metadata-only A100/data-host TODO

1. Inspect `meta/info.json`, `meta/tasks.jsonl`, one `meta/episodes/*.jsonl`, and
   one annotation record without fetching media or parquet shards where possible.
2. Confirm feature names, dtypes, shapes, sampling rate, task-index ordering,
   language annotations, skill IDs, skill boundary convention, and timestamps.
3. Inspect one small parquet row group only if needed to verify the 61-state and
   23-action arrays and their units; do not fetch the full dataset.
4. Instantiate R1Pro only on the A100 simulator host to assert `robot.action_dim`
   and actual flattened websocket keys against these static findings.
