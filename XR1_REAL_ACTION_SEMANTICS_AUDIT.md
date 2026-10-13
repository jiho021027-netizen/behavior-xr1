# XR-1 real action semantics audit

The checkpoint emits `(1,16,60)`; the stale external `(30,60)` client assertion is captured in `patches/xiaomi_runtime_dynamic_action_horizon.patch`. EEF translation uses each arm's own EEF matrix. EEF-local axis-angle vectors are converted with `R_base_from_eef @ r_local` before OmniGibson IK. Base, trunk, and gripper mappings remain blocked; no full-body behavior claim is made.
