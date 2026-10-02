# XR-1 arm rotation frame audit (offline)

- No Isaac process or `env.step` was run.
- Source: OmniGibson `InverseKinematicsController._update_goal` uses `dori @ R_current_base`.
- XR-1 source `recover_action` uses `R_current_world @ Exp(delta_local)`.
- Required conversion: `command_base = R_base_from_eef @ delta_local`.
- Candidate B matched both left and right saved poses to < 1e-17 rad; raw local candidate A was ~0.019995 rad wrong.
- Bridge patch applies this conversion to both arm rotation vectors; translation paths remain unchanged.
- Controller preprocessing contract: right converted command is within [-1, 1], so no clipping.
