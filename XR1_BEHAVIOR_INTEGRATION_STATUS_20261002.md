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
