# XR-1 to R1Pro frame mapping status

XR-1 action deltas are expressed in the current EEF orientation frame and recovered by `delta @ rotm.T`. OmniGibson `InverseKinematicsController(mode=pose_delta_ori)` accepts pose deltas in the robot-base frame according to `ik_controller.py::_update_goal`. A runtime bridge therefore needs an explicit EEF-to-base rotation transform.

Current status: **BLOCKED** until live custom R1Pro IK configuration and small-command validation are completed. No arbitrary 60D→23D mapping is implemented.
