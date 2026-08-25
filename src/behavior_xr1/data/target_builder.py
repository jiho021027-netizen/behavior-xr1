"""Source-bounded construction of provisional BEHAVIOR XR-1 training targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from behavior_xr1.geometry import Pose, quaternion_xyzw_to_matrix, relative_pose_delta

from .schema import BEHAVIOR_R1PRO_PROPRIO_DIM, XR1_ACTION_DIM


@dataclass(frozen=True)
class PackedAction:
    """60D values plus the dimensions whose semantics are currently known."""

    values: np.ndarray
    mask: np.ndarray


def _state(state: object, label: str) -> np.ndarray:
    array = np.asarray(state, dtype=np.float64)
    if array.shape != (BEHAVIOR_R1PRO_PROPRIO_DIM,):
        raise ValueError(f"{label} must have shape ({BEHAVIOR_R1PRO_PROPRIO_DIM},), got {array.shape}")
    return array


def behavior_action_mask() -> np.ndarray:
    """Known BEHAVIOR target dimensions; unknown/reserved dimensions remain false."""
    mask = np.zeros(XR1_ACTION_DIM, dtype=bool)
    mask[0:6] = True  # Left EEF local translation + axis-angle.
    mask[8:14] = True  # Right EEF local translation + axis-angle.
    mask[17:20] = True  # Source-defined R1Pro robot-local base velocity.
    mask[20:24] = True  # Direct finite difference of four trunk qpos values.
    return mask


def _eef_pose(state: np.ndarray, arm: str) -> Pose:
    if arm == "left":
        return Pose(state[17:20], quaternion_xyzw_to_matrix(state[20:24]))
    if arm == "right":
        return Pose(state[42:45], quaternion_xyzw_to_matrix(state[45:49]))
    raise ValueError(f"unknown arm {arm!r}")


def build_xr1_action_target(current_state: object, future_state: object) -> PackedAction:
    """Pack confirmed arm, base, and trunk targets from two 61D proprio rows.

    The base value is copied from ``future_state[0:3]`` because the pinned
    BEHAVIOR conversion utility associates action[t, 0:3] with the next
    state's recovered local base velocity. Actual LeRobot row alignment still
    requires data-host verification before training.
    """
    current = _state(current_state, "current_state")
    future = _state(future_state, "future_state")
    values = np.zeros(XR1_ACTION_DIM, dtype=np.float32)
    for arm, translation_slice, rotation_slice in (("left", slice(0, 3), slice(3, 6)), ("right", slice(8, 11), slice(11, 14))):
        delta = relative_pose_delta(_eef_pose(current, arm), _eef_pose(future, arm))
        values[translation_slice] = delta.translation
        values[rotation_slice] = delta.axis_angle
    values[17:20] = future[0:3]
    values[20:24] = future[53:57] - current[53:57]
    return PackedAction(values=values, mask=behavior_action_mask())


def build_xr1_action_horizon(current_state: object, future_states: object) -> PackedAction:
    """Build a target chunk from one current row and ``[H, 61]`` future rows."""
    current = _state(current_state, "current_state")
    futures = np.asarray(future_states, dtype=np.float64)
    if futures.ndim != 2 or futures.shape[1] != BEHAVIOR_R1PRO_PROPRIO_DIM:
        raise ValueError(f"future_states must have shape [H, {BEHAVIOR_R1PRO_PROPRIO_DIM}], got {futures.shape}")
    packed = [build_xr1_action_target(current, future) for future in futures]
    if not packed:
        return PackedAction(np.empty((0, XR1_ACTION_DIM), dtype=np.float32), np.empty((0, XR1_ACTION_DIM), dtype=bool))
    return PackedAction(np.stack([item.values for item in packed]), np.stack([item.mask for item in packed]))
