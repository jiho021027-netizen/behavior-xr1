"""Offline BEHAVIOR R1Pro -> Xiaomi XR-1 60D training conversion.

This module is deliberately simulator-free. It converts an already captured
trajectory mapping into the official generic XR-1 packed contract. It does
not create demonstrations or infer missing physical semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from behavior_xr1.geometry.pose import (
    Pose,
    axis_angle_from_rotation_matrix,
    relative_pose_delta,
)

MODEL_DIM = 60
STATE_DIM = 60
ACTION_HORIZON = 30
ACTION_ACTIVE_ARM_GRIPPER = np.array(
    [*range(0, 7), *range(8, 15)], dtype=np.int64
)
ACTION_ACTIVE_FULL = np.array(
    [*range(0, 7), *range(8, 15), 20, 21, 22, 23], dtype=np.int64
)
STATE_ACTIVE = np.arange(16, dtype=np.int64)

COMPACT_ACTION_XR1_INDICES = np.array([*range(0, 7), *range(8, 15), 20, 21, 22, 23], dtype=np.int64)
COMPACT_STATE_XR1_INDICES = np.array([*range(0, 15), 20, 21, 22, 23], dtype=np.int64)

def convert_native_episode(raw_state: object, raw_action: object) -> tuple[np.ndarray, np.ndarray]:
    """Canonical native R1Pro -> dense XR1 state/action conversion."""
    s = np.asarray(raw_state, dtype=np.float64)
    a = np.asarray(raw_action, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 256 or a.shape != (len(s), 23):
        raise ValueError(f"expected state (T,256), action (T,23), got {s.shape}, {a.shape}")
    if not np.isfinite(s).all() or not np.isfinite(a).all():
        raise ValueError("native streams contain non-finite values")
    from .r1pro_tiny_dataset import convert_recorded_trajectory
    arm_state, arm_action = convert_recorded_trajectory(s)
    state = np.zeros((len(s), MODEL_DIM), dtype=np.float64)
    state[:, :16] = arm_state[:, :16]
    state[:, 20:24] = s[:, 53:57]
    action = np.zeros((len(s), MODEL_DIM), dtype=np.float64)
    action[:-1, :15] = arm_action[:, :15]
    action[:, 20:24] = a[:, 3:7]
    return state, action



def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def axis_angle_to_rotation_matrix(axis_angle: object) -> np.ndarray:
    """Rodrigues exponential map for one 3-vector."""
    v = _array(axis_angle, (3,), "axis_angle")
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = v / theta
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def encode_arm_action(current: Pose, future: Pose) -> tuple[np.ndarray, np.ndarray]:
    """Encode the official XR-1 local EEF delta, independently per arm."""
    delta = relative_pose_delta(current, future)
    return delta.translation.copy(), delta.axis_angle.copy()


def reconstruct_arm_pose(current: Pose, translation: object, axis_angle: object) -> Pose:
    """Decode an XR-1 arm delta for numerical round-trip checks."""
    t = _array(translation, (3,), "translation")
    aa = _array(axis_angle, (3,), "axis_angle")
    return Pose(
        position=current.position + current.rotation @ t,
        rotation=current.rotation @ axis_angle_to_rotation_matrix(aa),
    )


def _required_trajectory_array(trajectory: Mapping[str, object], key: str, ndim: int) -> np.ndarray:
    if key not in trajectory:
        raise KeyError(f"trajectory is missing required field {key!r}")
    value = np.asarray(trajectory[key], dtype=np.float64)
    if value.ndim != ndim or value.shape[0] < 1:
        raise ValueError(f"trajectory[{key!r}] must be a non-empty rank-{ndim} array")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"trajectory[{key!r}] contains non-finite values")
    return value


def convert_state(trajectory: Mapping[str, object], timestep: int = 0) -> np.ndarray:
    """Pack R1Pro 7-DOF arm joints and aperture states into 60D XR-1 state."""
    left_joint = _required_trajectory_array(trajectory, "left_arm_joint", 2)
    right_joint = _required_trajectory_array(trajectory, "right_arm_joint", 2)
    left_gripper = _required_trajectory_array(trajectory, "left_gripper_aperture", 2)
    right_gripper = _required_trajectory_array(trajectory, "right_gripper_aperture", 2)
    n = left_joint.shape[0]
    if any(a.shape[0] != n for a in (right_joint, left_gripper, right_gripper)):
        raise ValueError("state fields must have equal trajectory length")
    if left_joint.shape[1] > 7 or right_joint.shape[1] > 7:
        raise ValueError("XR-1 state supports at most seven joints per arm")
    if not 0 <= timestep < n:
        raise IndexError(timestep)
    state = np.zeros((STATE_DIM,), dtype=np.float64)
    state[: left_joint.shape[1]] = left_joint[timestep]
    state[7] = left_gripper[timestep, 0]
    state[8 : 8 + right_joint.shape[1]] = right_joint[timestep]
    state[15] = right_gripper[timestep, 0]
    return state


def _pose_series(trajectory: Mapping[str, object], side: str) -> list[Pose]:
    positions = _required_trajectory_array(trajectory, f"{side}_ee_position", 2)
    rotations = _required_trajectory_array(trajectory, f"{side}_ee_rotation", 3)
    if positions.shape[1] != 3 or rotations.shape[1:] != (3, 3) or positions.shape[0] != rotations.shape[0]:
        raise ValueError(f"{side} EEF position/rotation arrays have incompatible shapes")
    return [Pose(positions[i], rotations[i]) for i in range(positions.shape[0])]


def convert_action(trajectory: Mapping[str, object], timestep: int, target_timestep: int) -> np.ndarray:
    """Pack one future R1Pro target into an unnormalized 60D XR-1 action."""
    left = _pose_series(trajectory, "left")
    right = _pose_series(trajectory, "right")
    if not (0 <= timestep < len(left) and 0 <= target_timestep < len(left)):
        raise IndexError((timestep, target_timestep))
    out = np.zeros((MODEL_DIM,), dtype=np.float64)
    lt, lr = encode_arm_action(left[timestep], left[target_timestep])
    rt, rr = encode_arm_action(right[timestep], right[target_timestep])
    out[0:3], out[3:6] = lt, lr
    out[8:11], out[11:14] = rt, rr
    for index, key in ((6, "left_gripper_aperture"), (14, "right_gripper_aperture")):
        # A real recording may expose one scalar or two physical finger qpos.
        # The generic XR-1 field is scalar; averaging the two recorded qpos is
        # deterministic and preserves the source signal without inventing an
        # aperture calibration.
        if key in trajectory:
            values = _required_trajectory_array(trajectory, key, 2)
            scalar = values.mean(axis=1)
        else:
            scalar_key = key.replace("_aperture", "_scalar")
            values = _required_trajectory_array(trajectory, scalar_key, 2)
            scalar = values[:, 0]
        out[index] = scalar[target_timestep] - scalar[timestep]
    if "base_velocity" in trajectory:
        base = _required_trajectory_array(trajectory, "base_velocity", 2)
        if base.shape[1] != 3 or base.shape[0] <= target_timestep:
            raise ValueError("base_velocity must be (T,3)")
        out[17:20] = base[target_timestep]
    if "waist_delta" in trajectory:
        waist = _required_trajectory_array(trajectory, "waist_delta", 2)
        if waist.shape[1] != 1 or waist.shape[0] <= target_timestep:
            raise ValueError("waist_delta must be (T,1)")
        out[16] = waist[target_timestep, 0]
    return out


def build_action_chunk(trajectory: Mapping[str, object], timestep: int, horizon: int = ACTION_HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """Build a horizon chunk; tail targets repeat the final observed pose."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    length = _pose_series(trajectory, "left").__len__()
    chunk = np.zeros((horizon, MODEL_DIM), dtype=np.float64)
    mask = np.zeros((horizon, MODEL_DIM), dtype=np.float64)
    for i in range(horizon):
        target = min(timestep + i + 1, length - 1)
        chunk[i] = convert_action(trajectory, timestep, target)
        mask[i, ACTION_ACTIVE_ARM_GRIPPER] = 1.0
        if "base_velocity" in trajectory:
            mask[i, 17:20] = 1.0
        if "waist_delta" in trajectory:
            mask[i, 16] = 1.0
    return chunk, mask


def compute_normalization(samples: Sequence[Mapping[str, np.ndarray]], action_mask: np.ndarray, state_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-step action and state statistics using Xiaomi's epsilon-ready convention."""
    if not samples:
        raise ValueError("at least one converted sample is required")
    actions = np.concatenate([np.asarray(s["actions"], dtype=np.float64) for s in samples], axis=0)
    states = np.concatenate([np.asarray(s["states"], dtype=np.float64) for s in samples], axis=0)
    if actions.ndim != 2 or actions.shape[1] != MODEL_DIM:
        raise ValueError("actions must concatenate to (N,60)")
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError("states must concatenate to (N,60)")
    am = np.asarray(action_mask, dtype=bool).reshape(MODEL_DIM)
    sm = np.asarray(state_mask, dtype=bool).reshape(STATE_DIM)
    mean, std = actions.mean(0), actions.std(0)
    state_mean, state_std = states.mean(0), states.std(0)
    mean[~am] = std[~am] = 0.0
    state_mean[~sm] = state_std[~sm] = 0.0
    return {"action_mean": mean, "action_std": std, "state_mean": state_mean, "state_std": state_std, "action_mask": am.astype(np.int32), "state_mask": sm.astype(np.int32)}
