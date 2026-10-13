"""Adapter for the live 2026 R1Pro 61D proprio observation."""
from __future__ import annotations
import numpy as np

RAW_DIM=61; XR1_DIM=60
# Verified flatten order: configs/eval/r1pro_xr1_ik.yaml:13-28 and
# OmniGibson robots/robot.py get_proprioception concatenation.
SLICES={
 'base_qvel':slice(0,3), 'left_arm_qpos':slice(3,10), 'left_arm_qvel':slice(10,17),
 'left_eef_pos':slice(17,20), 'left_eef_quat':slice(20,24),
 'left_gripper_qpos':slice(24,26), 'left_gripper_qvel':slice(26,28),
 'right_arm_qpos':slice(28,35), 'right_arm_qvel':slice(35,42),
 'right_eef_pos':slice(42,45), 'right_eef_quat':slice(45,49),
 'right_gripper_qpos':slice(49,51), 'right_gripper_qvel':slice(51,53),
 'trunk_qpos':slice(53,57), 'trunk_qvel':slice(57,61),
}

def adapt_r1pro_2026_state(proprio) -> np.ndarray:
    x=np.asarray(proprio,dtype=np.float32).reshape(-1)
    if x.shape != (RAW_DIM,): raise ValueError(f'expected ({RAW_DIM},), got {x.shape}')
    if not np.isfinite(x).all(): raise ValueError('2026 proprio contains non-finite values')
    y=np.zeros(XR1_DIM,dtype=np.float32)
    y[0:7]=x[SLICES['left_arm_qpos']]; y[8:15]=x[SLICES['right_arm_qpos']]
    # Existing XR-1 state contract uses scalar gripper channels; official qpos
    # vectors are reduced by their mean, matching the validated 2025 reducer.
    y[7]=np.mean(x[SLICES['left_gripper_qpos']]); y[15]=np.mean(x[SLICES['right_gripper_qpos']])
    # Frozen XR-1 dense trunk destination. Base velocity has no verified
    # destination in the current 60D state contract and remains reserved.
    y[20:24]=x[SLICES['trunk_qpos']]
    if not np.isfinite(y).all(): raise AssertionError('adapted state non-finite')
    return y
