# 2026 R1Pro → XR-1 state contract handoff

The live Task 0 fixture is `artifacts/2026_state_validation/task0_reset_observation.npz`. Its `.proprio` is finite `float32 (61,)` and its task id is `0`.

The verified source concatenation is:

`base_qvel[0:3]`, `left_arm_qpos[3:10]`, `left_arm_qvel[10:17]`, `left_eef_pos[17:20]`, `left_eef_quat[20:24]`, `left_gripper_qpos[24:26]`, `left_gripper_qvel[26:28]`, `right_arm_qpos[28:35]`, `right_arm_qvel[35:42]`, `right_eef_pos[42:45]`, `right_eef_quat[45:49]`, `right_gripper_qpos[49:51]`, `right_gripper_qvel[51:53]`, `trunk_qpos[53:57]`, `trunk_qvel[57:61]`.

The adapter at `src/behavior_xr1/data/r1pro_2026_state.py` emits dense `float32 (60,)`: left arm qpos → `[0:7]`, right arm qpos → `[8:15]`, gripper pairwise means → `[7]` and `[15]`, trunk qpos → `[20:24]`; all other destinations are explicitly zero/reserved. Direct numerical differences on the captured fixture are zero.

Gripper conversion is **PARTIAL**: pairwise mean is the implemented reducer and matches the captured values, but authoritative physical units, opening direction, and limits are not established. Base velocity, EEF pose, joint velocities, and remaining dense channels are intentionally unused pending a contract decision.

`TRAIN_EVAL_STATE_CONTRACT_MATCH=PENDING_TEAM_VERIFICATION`. The training adapter must consume this exact schema and confirm gripper/base/reserved semantics before claiming parity. Normalization must be supplied separately for the 2026 contract; 2025 statistics are not assumed compatible.
