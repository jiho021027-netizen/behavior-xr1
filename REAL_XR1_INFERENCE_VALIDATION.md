# Real XR-1 inference validation

`REAL_XR1_MODEL_LOAD_VALIDATED=True`; `REAL_XR1_FORWARD_VALIDATED=True`. The local checkpoint produced `(1,16,60)` and action index 0 is the first temporal action. The external Xiaomi client still needs the documented dynamic-horizon patch.

The BEHAVIOR cross-embodiment state, camera, task-conditioning, base, trunk, and gripper mappings are not fully source-validated. No full-body behavior step or task success is claimed.
