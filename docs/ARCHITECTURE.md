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

## Verified BEHAVIOR Action Representation

Verification run on 2026-08-24 was intentionally metadata-first. No local
2026 LeRobot dataset, `meta/info.json`, `meta/tasks.jsonl`, `meta/stats.json`,
or BEHAVIOR parquet shard was found under `/home/kang` or the mounted Windows
`C:` drive. The installed Python environment has no `omnigibson`, `torch`, or
`lerobot` module. No package, asset, checkpoint, or dataset was installed or
downloaded. Consequently, every item in this section is either source-backed
or explicitly unavailable at runtime.

### Demonstration Provenance

**Source-backed:** `docs/challenge/dataset.md` specifies 2026 demos as a
LeRobot v3 dataset. `OmniGibson/scripts/learning/update_lerobot_base_qvel.py`
uses default `--action-width 23`, reads raw HDF5 `demo["action"]`, and compares
it to the parquet `action` column within configurable tolerance. It compares
`action[t, 0:3]` to the next frame's recovered robot-local base velocity. This
confirms the repository's expected 23D stored action width and first-three
base convention; it does **not** prove a per-episode controller-config record.

**Actual-data verification:** NOT AVAILABLE. Feature schema, dtype, timestamp,
task-index, episode/frame-index, skill fields, and the exact contents of a
demo action/state row require an existing local dataset or a future
metadata-only data-host inspection.

### Offline EE Target Reconstruction

`eval_utils.py::PROPRIOCEPTION_INDICES["R1Pro"]` defines the presumed 61D
layout: base local velocity `0:3`; left arm qpos `3:10`, qvel `10:17`, EEF
position `17:20`, EEF quaternion `20:24`, gripper qpos `24:26`, gripper qvel
`26:28`; right equivalents qpos `28:35`, qvel `35:42`, EEF position `42:45`,
EEF quaternion `45:49`, gripper qpos `49:51`, gripper qvel `51:53`; trunk qpos
`53:57`, trunk qvel `57:61`.

If actual `observation.state` uses this source layout, EEF pose is directly
stored, so FK is not necessary for the arm delta target. Given current pose
`T_t` and future pose `T_{t+i}`, local relative motion is deterministically
computable. If the live/demo schema differs or lacks EEF pose, R1Pro assets and
qpos could in principle support FK, but that has not been verified.

### Live R1Pro IK Controller

**NOT AVAILABLE:** this host has no pre-existing OmniGibson runtime or assets.
No smoke initialization was attempted, so robot class, loaded controller order,
controller classes/command dimensions, EEF links, IK modes, and live
`robot.action_dim` are unverified. The 21D total remains a source-derived
arithmetic inference only.

### XR-1 → R1Pro Frame Conversion

XR-1 `JsonDataset._arm_action()` creates translation
`R_t^T(p_{t+i}-p_t)` and rotation `axis_angle(R_t^T R_{t+i})`; this is the
local EEF convention equivalent to the relative transform
`inverse(T_t) @ T_{t+i}`. `recover_action()` applies the inverse composition.

OmniGibson IK `pose_delta_ori` instead receives translation relative to the
robot base and composes orientation relative to the current EEF pose. Therefore
the future adapter must transform XR-1 translation by the current EEF rotation
into base coordinates before issuing an IK command. Its full live correctness
is unverified until a minimal R1Pro smoke test.

### Remaining Embodiment Mismatches

- XR-1 waist is one relative scalar (`16:17`); R1Pro default trunk is four
  absolute joint-position commands (`3:7`). No source-backed mapping exists.
- XR-1 gripper is one relative scalar per hand; R1Pro smooth gripper consumes
  one controller command that produces an absolute two-finger target. Exact
  limits and relative-to-absolute rule require asset/runtime evidence.
- XR-1 EE actions are local-EEF; BEHAVIOR IK commands are robot-base-relative.
- XR-1 base velocity frame/unit is not documented in XR-1 source.

### Final Action-Space Decision

**Strategy B: BLOCKED.** The current source audit makes it a plausible
candidate, but neither acceptance condition has runtime/data confirmation:

1. actual demo state/action schema needed to establish deterministic target
   reconstruction; and
2. live R1Pro IK load/action-dimension/frame behavior.

Do not implement action conversion or select Strategy B training targets until
both checks are complete on an A100/data host. Keep Strategy A only as the
documented source-compatible baseline, not as a conversion implementation.

## Metadata-only A100/data-host TODO

1. Inspect `meta/info.json`, `meta/tasks.jsonl`, one `meta/episodes/*.jsonl`, and
   one annotation record without fetching media or parquet shards where possible.
2. Confirm feature names, dtypes, shapes, sampling rate, task-index ordering,
   language annotations, skill IDs, skill boundary convention, and timestamps.
3. Inspect one small parquet row group only if needed to verify the 61-state and
   23-action arrays and their units; do not fetch the full dataset.
4. Instantiate R1Pro only on the A100 simulator host to assert `robot.action_dim`
   and actual flattened websocket keys against these static findings.

## BEHAVIOR-Specific XR-1 60D Packing

This is a provisional, training-side representation implemented in
`behavior_xr1`. It does not change either upstream repository and it is not a
runtime XR-1-to-OmniGibson controller adapter.

### Why Reserved Dimensions Are Reused

**PARTIAL acceptance.** In `xr1/mibot/utils/io.py`, `compose_action()` creates
a 60D zero vector and writes only `ACTION_PARTS`; `build_action_mask()` makes
only those listed parts valid. `compose_state()` likewise writes a 60D zero
vector, filling only arm and gripper values. Conversely,
`xr1/mibot/models/VLA/XR1.py::_build_model()` constructs 60-wide state/action
projectors and a 60-wide action output; `dit_forward()` accepts the action mask
and `compute_flow_loss()` indexes masked dimensions. Therefore `20:60` is not
architecturally hard-coded to zero and an arbitrary boolean dimension mask can
reach the flow loss.

The dimensions were nevertheless zero/masked throughout the released XR-1
data pipeline, so their pretrained weights/statistics have no established
BEHAVIOR semantics. Reuse is architecture-compatible, but training-scale
normalization and effectiveness remain unverified.

### State Packing

`adapters/state.py::BehaviorStatePacker` accepts `[...,61]` following
`PROPRIOCEPTION_INDICES["R1Pro"]` and emits `[...,60]`:

| Output slice | Source slice | Status |
|---|---|---|
| `0:7` | left arm qpos `3:10` | source-backed |
| `7:8` | caller supplied scalar from left gripper qpos `24:26` | explicit callback required |
| `8:15` | right arm qpos `28:35` | source-backed |
| `15:16` | caller supplied scalar from right gripper qpos `49:51` | explicit callback required |
| `16:20` | trunk qpos `53:57` | source-backed four-value storage; joint identities unavailable |
| `20:23` | robot-local base qvel `0:3` | source-backed |
| `23:60` | zero | intentionally reserved |

No gripper reducer means `NotImplementedError`; no average/finger selection is
invented. This preserves the native XR-1 arm/gripper prefix while using its
previously zero state tail for R1Pro trunk/base information.

### Action Packing

`data/target_builder.py::build_xr1_action_target` forms one provisional 60D
target from current/future 61D proprio rows. EEF positions and xyzw
quaternions are robot-base-relative: `Robot._get_proprioception_dict()` calls
`get_relative_eef_pose()`, which uses `relative_pose_transform(eef, base)`.
The pose math exactly uses XR-1 `JsonDataset._arm_action()`:
`R_t.T @ (p_future-p_t)` and `axis_angle(R_t.T @ R_future)`.

| Output slice | Value | Valid now? |
|---|---|---|
| `0:3`, `3:6` | left local EEF translation, axis-angle | yes |
| `6:7`, `7:8` | left gripper / reserved | no |
| `8:11`, `11:14` | right local EEF translation, axis-angle | yes |
| `14:16` | right gripper / reserved | no |
| `16:17` | XR-1 waist | no: R1Pro trunk is 4D, no collapse |
| `17:20` | future-row R1Pro local base qvel | source-backed temporal convention; dataset alignment pending |
| `20:24` | future-current trunk qpos delta | yes as a numerical target; physical joint semantics pending |
| `24:60` | reserved | no |

### Training Target Reconstruction and Validity Mask

`behavior_action_mask()` enables only `0:6`, `8:14`, and `17:24`; its gripper,
waist, and all remaining reserved dimensions are false. `PackedAction` carries
the values and mask together, so unavailable semantics cannot silently become
zero-valued loss targets. A future XR-1 compatibility wrapper should pass this
mask to the existing `XR1.dit_forward()` / `compute_flow_loss()` action-mask
path; upstream code is unchanged in this milestone.

### Runtime Adapter Still Blocked

No `XR1ActionToBehaviorAction` logic has been added. The live IK controller,
XR-1 base frame/unit, gripper conversion, and 1D-waist-to-4D-trunk mapping are
still unverified. In particular, this training target does not authorize
zero-padding, broadcasting, or issuing a 21D OmniGibson action.

## Pretrained-Safe BEHAVIOR Dimension Extension

### Problem with Previously-Zero Dimensions

At Xiaomi-Robotics-1 commit `556cca33963a2b36d835a40374c3b4c8eef68401`,
`xr1/mibot/utils/io.py::compose_state` only writes state `0:16`; state `16:60`
is zero. `XR1.py::Projector._init_weights` randomly initializes every linear
weight, so a checkpoint column connected only to a zero feature receives no
data-driven update. Activating such a column later changes the condition by
an uncontrolled pretrained-random term.

The action case is narrower: `utils/io.py::ACTION_PARTS` includes base `17:20`,
so this project must preserve that pretrained action region. This contract
only treats trunk-action `20:24` as new. Whether any released checkpoint
actually trained each supported action part remains checkpoint/data dependent.

### State Extension

The proposed state tail is trunk qpos `16:20` and robot-local base qvel
`20:23`; see `BehaviorStatePacker`. They are passed only after obtaining
dataset-fitted state quantiles. The opt-in `r1pro_gripper_width_sum` uses the
2025 winner's two-finger sum, but remains non-default until 2026 rows verify
the pair relation.

### Action Extension

The 60D target keeps native EEF regions and the established base `17:20`.
Four trunk qpos finite differences occupy `20:24` with their own validity
mask. They are not a live controller mapping: trunk joint identity/order,
episode alignment, and controller acceptance remain A100 checks.

### Initialization Strategy

**Recommended: targeted zero initialization (Strategy 2).** Immediately after
strict checkpoint loading and before optimizer construction, zero only:

```python
# PyTorch Linear weight convention is [out_features, in_features].
with torch.no_grad():
    model.state_projector.layers[0].weight[:, 16:23] = 0
    model.state_projector_choice.layers[0].weight[:, 16:23] = 0
    model.action_projector.layers[0].weight[:, 20:24] = 0
    model.action_output_layer.layers[2].weight[20:24, :] = 0
    for choice in range(5):
        model.action_projector_choice[1].layers[0].weight[choice * 60 + 20:choice * 60 + 24, :] = 0
```

These are exact names from `XR1.py::_build_model` and `Projector`: each first
projector linear is `layers.0`; the two-layer action output's final linear is
`layers.2`. Their default `bias=False`, so no corresponding bias exists. The
choice path is included because it consumes state during training and predicts
five flattened 60D alternatives. The NumPy-only shape-checked mirror is
`models/initialization.py::zero_new_behavior_parameters`.

| Exact `named_parameters()` name | Shape from source | BEHAVIOR operation |
|---|---:|---|
| `state_projector.layers.0.weight` | `[1024, 60]` | zero input columns `16:23` |
| `state_projector_choice.layers.0.weight` | `[vlm.config.text_config.hidden_size, 60]` | zero input columns `16:23` |
| `action_projector.layers.0.weight` | `[1024, 60]` | zero input columns `20:24` |
| `action_output_layer.layers.2.weight` | `[60, 1024]` | zero output rows `20:24` |
| `action_projector_choice.1.layers.0.weight` | `[300, vlm.config.text_config.hidden_size]` | zero rows `60*c + 20:60*c + 24`, `c=0..4` |

There are no biases for these five `Projector` instances: `Projector` defaults
to `bias=False`. The VLM hidden width is loaded from the external Qwen config,
so source alone does not establish a numeric value without loading that model.

Strategy 1 is simplest but violates preservation by immediately admitting
untrained columns. A zero-init residual adapter (Strategy 3) is also
preserving and parameter-efficient, but adds checkpoint keys, an integration
site, and an ablation dimension before the minimal extension has been tested.
Strategy 2 preserves the native path exactly at initialization, retains strict
checkpoint compatibility, and learns only from masked BEHAVIOR losses; use an
adapter only if its ablation materially improves this baseline.

### Normalization Strategy

XR-1 `JsonDataset.__getitem__` first builds raw parts, normalizes actions with
per-step `mean[H,60]`/`std[H,60]`, quantile-normalizes state using
`q01[1,60]`/`q99[1,60]`, then builds the action mask. In `XR1.dit_forward`,
the mask is applied before the action projector; `compute_flow_loss` indexes
only masked values. Thus new dimensions require real-data statistics before
they become valid. Identity normalization is acceptable only as an explicit
temporary inactive schema (`mean=0`, `std=1`, invalid mask); it is not a
training substitute.

The A100 dataset job must emit `BehaviorNormStats`: `state_q01[1,60]`,
`state_q99[1,60]`, `action_mean[H,60]`, `action_std[H,60]`, and
`action_valid_mask[H,60]`. Populate native dimensions with the selected
checkpoint-compatible stats; calculate the new state/action slices from the
confirmed BEHAVIOR transform and horizon alignment. No values are fabricated
locally.

### 2025 Winner Reference

At `behavior-1k-solution-2025` commit
`ca556f74a455cef7987a2be4537b5ac85cc56dd7`,
`b1k_policy.py::extract_state_from_proprio` emits 23D
`[base(3), trunk(4), left-arm(7), left-width(1), right-arm(7), right-width(1)]`.
Each width is `sum(two finger qpos)` mapped from `[0,0.1]` to `[-1,1]`.
`DataConfigFactory.create` uses `DeltaActions` with boolean mask
`make_bool_mask(-3,3,-1,7,-1,7,-1)`, passed verbatim to OpenPI's
`DeltaActions`; its implementation is an external dependency and is not
vendored in the winner checkout, so this document does not assign unverified
per-coordinate delta/absolute meaning beyond the source mask. The winner's
statistics/tokenizer scripts do establish that every horizon action is
transformed relative to the one current state. `PadStatesAndActions(32)` pads
their 23D state/actions to the model's 32D space, and `B1kOutputs` truncates
inference back to 23D. This is a representation reference, not evidence that
2026 dataset semantics or an XR-1 checkpoint are identical.
