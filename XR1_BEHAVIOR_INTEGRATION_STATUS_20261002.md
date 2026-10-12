# XR-1 × BEHAVIOR Integration Status

## 1. Executive Summary

The XR-1 checkpoint loads and produces a real `(1, 16, 60)` action chunk. The corrected EEF translation and axis-angle frame conversions are source-applied and offline validated. Arm-only runtime reached `env.step`, but full-body execution and task success remain blocked.

## 2. Real XR-1 Model Validation

`REAL_XR1_MODEL_LOAD_VALIDATED=True`; `REAL_XR1_FORWARD_VALIDATED=True`. The first temporal action is index 0. No padding or interpolation is used.

## 3. BEHAVIOR -> XR-1 Input Adaptation

The current state/camera/gripper mappings are adaptation assumptions. Full semantic equivalence is not validated.

## 4. Action Chunk Contract

The checkpoint emits `(H,60)` with `H=16`. The stale Xiaomi client `(30,60)` assertion is documented in `patches/xiaomi_runtime_dynamic_action_horizon.patch`; this external patch is not silently treated as a behavior-xr1 source change.

## 5. EEF Translation Frame Calibration

`R_base_from_eef = R_world_from_base.T @ R_world_from_eef`; `v_base = R_base_from_eef @ v_eef`. Left error is approximately `3.45e-09`; right error after using its own EEF matrix is approximately `3.5e-09`.

## 6. Axis-Angle Rotation Frame Calibration

XR-1 uses `R_target_world = R_current_world @ Exp(delta_local)`. OmniGibson IK uses `R_target_base = Exp(command_base) @ R_current_base`. The bridge now applies `r_base = R_base_from_eef @ r_local` for both arms. Offline basis and saved-action SO(3) errors are below `1.2e-16` rad.

## 7. R1Pro Controller Contract

R1Pro has four physical torso joints. Arm IK command dimensions are six per arm; controller preprocessing accepts the validated converted commands.

## 8. Correct Hold / No-op Construction

Smooth grippers require controller input `1.0` for the validated hold. Corrected hold drift was `3.76e-07` left and `2.24e-07` right.

## 9. Real Arm-only Physics Smoke Test

A real arm-only step reached the environment successfully. It predates the final frame/rotation corrections, so `REAL_XR1_ARM_ONLY_SEMANTIC_RESPONSE_VALIDATED=PARTIAL`.

## 10. Remaining Full-body Blockers

`BASE_ACTION_MAPPING_VALID=False`, `TRUNK_ACTION_MAPPING_VALID=False`, and `GRIPPER_ACTION_MAPPING_VALID=False`. Consequently `FULL_REAL_XR1_BEHAVIOR_STEP_VALIDATED=False`, `REAL_XR1_SHORT_CLOSED_LOOP_VALIDATED=False`, and `TASK_SUCCESS_VALIDATED=False`.

## 11. TRUE-null Study Correction

The historical ~26.55x read-only gain was a `PREPATCH_FRAME_ARTIFACT`. Corrected Run2 Q0 gain was `0.0015303294457449821`, Run2 gain `0.0013567822397314575`, ratio `0.8865948724334715`. Matched corrected-frame live improvement was zero. No task-performance improvement is claimed.

## 12. Validated Claims

Model forward, dynamic action horizon, independent EEF translation frames, EEF-local-to-base rotation conversion, corrected hold construction, and the existence of an arm-only physics path are validated at their stated scopes.

## 13. Claims Not Yet Validated

Full-body XR-1 action semantics, short closed-loop behavior, and task success are not validated.

## 14. Next Work

Reconcile base, four-joint trunk, and gripper mappings from source contracts, then perform a fresh corrected matched arm validation before any broader live test.

## Base action mapping audit (2026-10-06)

Offline source/artifact audit completed; no Isaac runtime or `env.step` was executed. XR-1 `ACTION_PARTS` proves only base indices 17:20 in `[x,y,rz]` order. The inspected Xiaomi source does not provide base execution frame, units, control frequency, or a deployment equation. R1Pro is `HolonomicBaseJointController` velocity `[x,y,rz]`, local-frame linear input, input limits [-1,1], output limits [-0.75,-0.75,-1]..[0.75,0.75,1], with linear x/y rotated to canonical coordinates. Identity remains an unresolved candidate only.

`BASE_ACTION_MAPPING_VALID=False`; `BASE_LIVE_TEST_EXECUTED=False`; `REAL_XR1_BASE_ONLY_ENV_STEP_RUNTIME_VALIDATED=False`. See `XR1_BASE_ACTION_MAPPING_AUDIT.md` and `xr1_base_action_mapping_audit.json`.

## Gripper action mapping audit (2026-10-06)

XR-1 gripper slots are action indices 6 and 14. `recover_action` proves relative accumulation (`g_next = g_current + delta`), but the checked-in XR-1 source does not define the scalar's physical meaning, range, endpoints, sign, clipping, units, or source execution consumer. R1Pro is fully closed: smooth scalar `u` clips to [-1,1], broadcasts to both fingers, and maps to `0.025*(u+1)` m in [0,0.05]. The exact saved output is left delta `0.9921884536743164`, right delta `8.164062705873221e-07`; no physical cross-embodiment conversion can be justified.

`GRIPPER_SOURCE_CONTRACT_UNAVAILABLE=True`; `GRIPPER_ACTION_MAPPING_VALID=False`; `GRIPPER_LIVE_TEST_EXECUTED=False`. See `XR1_GRIPPER_ACTION_MAPPING_AUDIT.md` and `xr1_gripper_action_mapping_audit.json`.

## Trunk action mapping audit (2026-10-06)

XR-1 source proves waist action index 16 and relative accumulation against optional `robot_state[waist_pos]`. It does not provide a waist state index in `compose_state` (the optional waist field is omitted), physical coordinate meaning, range, units, sign, clipping, source geometry, or reference initialization. R1Pro uses absolute four-joint `JointController` targets for torso joints 1–4. JoyLo contains an R1Pro-only `trunk_translate` manifold, but no evidence connects it to XR-1 waist semantics.

`TRUNK_SOURCE_CONTRACT_UNAVAILABLE=True`; `TRUNK_ACTION_MAPPING_VALID=False`; no live test was run. See `XR1_TRUNK_ACTION_MAPPING_AUDIT.md` and `xr1_trunk_action_mapping_audit.json`.

## Checkpoint embodiment audit (2026-10-06)

Offline checkpoint inventory found one local Xiaomi checkpoint, `Xiaomi-Robotics-1-RoboCasa365`; its `(16,60)` normalization mask activates only `0:12`, so it is a single-EEF RoboCasa365 contract and cannot supply right-arm fields. The official source code documents a generic bimanual 60D packed schema, and the public `XiaomiRobotics/Xiaomi-Robotics-1-5B` foundation checkpoint is a generic post-training candidate, but no local or public R1Pro-specific checkpoint contract was proven. No download, inference, simulator, GPU, or `env.step` was run.

`BIMANUAL_SCHEMA_SUPPORTED_BY_CODE=True`; `BIMANUAL_CHECKPOINT_FOUND=True` (public generic candidate); `BIMANUAL_CHECKPOINT_LOCAL=False`; `BIMANUAL_CHECKPOINT_CONTRACT_PROVEN=False`; `R1PRO_EMBODIMENT_CONTRACT_PROVEN=False`. Historical assumed-layout arm runtime and frame-transform claims remain qualified and preserved; they do not establish canonical bimanual semantics. See the corresponding checkpoint embodiment audit artifacts from the prior audit workspace.

## Official Xiaomi-Robotics-1-5B contract audit (2026-10-06)

The public 5B foundation release is proven to use the official generic bimanual 60D packed schema: left EEF `0:6`, left gripper `6:7`, right EEF `8:14`, right gripper `14:15`, waist `16:17`, and base `17:20`. Its generic source confirms EEF-local relative position deltas and axis-angle rotation composition. The runtime and post-training configuration keep normalization statistics external; no R1Pro-specific robot type, action/state statistics, physical gripper contract, base contract, waist contract, or canonical replan interval is released.

`XR1_5B_BIMANUAL_CONTRACT_VALID=True`; `XR1_5B_R1PRO_DIRECT_CONTRACT_VALID=False`; `XR1_5B_REQUIRES_R1PRO_POSTTRAINING=True`; `XR1_5B_ZERO_SHOT_R1PRO_FEASIBILITY=REQUIRES_R1PRO_NORMALIZATION_AND_POSTTRAINING`. `ROBOCASA365_CHECKPOINT_BIMANUAL=False` remains unchanged. The existing EEF frame transform remains reusable only for the proven arm delta fields. No 5B weights, inference, simulator, GPU, or `env.step` were run. See `XR1_5B_BIMANUAL_CONTRACT_AUDIT.md`, `xr1_5b_bimanual_contract_audit.json`, `xr1_5b_robot_type_inventory.json`, and `xr1_5b_canonical_action_schema.json`.

## R1Pro post-training data pipeline audit (2026-10-06)

A simulator-free BEHAVIOR_R1PRO converter, horizon-30 chunk builder, and R1Pro-only normalization utility are implemented and synthetic round-trip validated. The repository contains no expert/demo trajectory bundle: BEHAVIOR validation data is scene/task metadata and assets only. Consequently `R1PRO_POSTTRAIN_DATA_CONTRACT_VALID=True` but `R1PRO_REAL_POSTTRAIN_DATA_READY=False`, `REAL_BEHAVIOR_EPISODE_CONVERSION_VALID=False`, and `POSTTRAIN_RECOMMENDATION=NO_GO_NO_DEMONSTRATIONS`.

The initial mask is dual arms plus grippers (`0:7`, `8:15`); base `17:20` and waist `16` remain masked until real trajectory coverage and physical semantics are documented. See `XR1_R1PRO_POSTTRAIN_DATA_AUDIT.md`, `xr1_r1pro_posttrain_data_audit.json`, `r1pro_xr1_training_action_schema.json`, `r1pro_xr1_training_state_schema.json`, and `r1pro_xr1_normalization_spec.json`. Full suite: 62 passed, 0 failed.

## Official R1Pro demonstration validation (2026-10-06)

Downloaded one official episode from `behavior-1k/2025-challenge-demos` (`task-0000/episode_00000010`, `turning_on_radio`). Dataset metadata identifies `robot_type=R1Pro`; the selected payload is 8,469,393 bytes with 1,956 synchronized frames at 30 Hz, 256D state, 23D recorded controller action, instruction annotation, and readable head/left-wrist/right-wrist RGB videos. All 1,955 transitions converted through the R1Pro-to-XR1 EEF equations. Maximum translation reconstruction error was `1.11e-16`; maximum SO(3) matrix error was `1.96e-15` left and `1.83e-15` right. Horizon-30 `(30,60)` sample and validation-only statistics were generated.

`R1PRO_REAL_SAMPLE_PIPELINE_VALID=True`; `R1PRO_FULL_TRAINING_DATA_READY=False`; `TINY_OVERFIT_DATA_READY=False`. The one-episode sample proves real decoding and conversion plumbing, not production data coverage or final normalization statistics. Initial action mask remains dual arms plus grippers; base and waist remain masked. Full suite: 65 passed, 0 failed. See `XR1_R1PRO_REAL_DEMO_VALIDATION.md`, `xr1_r1pro_real_demo_validation.json`, and `behavior_demo_validation_manifest.json`.

## 2026-10-06: minimal real R1Pro tiny dataset

Validated three official R1Pro episodes (17,568 frames; 384 deterministic, episode-bounded horizon-30 samples) for a future 5B training smoke test. The source-stream conversion, RGB/state/action batch assembly, and masks pass; no 5B weights, GPU inference, or training was run. See `XR1_R1PRO_TINY_DATASET_AUDIT.md`.

## 2026-10-06: XR1 5B tiny-overfit attempt

Downloaded and integrity-verified the official 10.23 GB Xiaomi-Robotics-1-5B checkpoint. The real model could be constructed but failed at GPU transfer with NVIDIA `NV_ERR_NO_MEMORY`; no forward, backward, optimizer update, or training occurred. `XR1_5B_R1PRO_TRAINING_STACK_VALID=False`. See `XR1_5B_R1PRO_TINY_OVERFIT.md`.

## 2026-10-06: zero-shot radio diagnostic placement gate

The requested one-attempt low-memory meta/staged XR1-5B placement failed at meta construction (`Tensor.item()` on a meta tensor). Hard Gate A prevented any inference, simulator task, action execution, or video recording. `GB10_XR1_5B_MODEL_PLACEMENT_VALID=False`. See `XR1_5B_R1PRO_ZERO_SHOT_RADIO.md`.

## 2026-10-06: meta-compatible XR1 constructor and staged placement

Patched the XR1 initialization-only Beta distribution to remain CPU-resident during meta construction. CPU structure/state equivalence passed and the meta model built successfully. A single BF16 meta/to_empty/memory-mapped staged-copy placement completed all checkpoint copies without an allocation error; future persistent loaders must call `tie_weights()` after `to_empty` to restore Qwen alias identity. No forward, training, data, or simulator was run.

## 2026-10-06: first real offline XR1/R1Pro forward

The model loaded and tied weights were restored. One real R1Pro `turning_on_radio` observation reached CUDA inference, which failed before output because NVRTC rejected the GPU architecture flag. Hard Gate 1 prevented simulator launch, actions, task evaluation, and video.
