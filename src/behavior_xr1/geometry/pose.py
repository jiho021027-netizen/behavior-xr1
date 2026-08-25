"""Small NumPy pose operations matching XR-1 JsonDataset._arm_action().

All rotations are 3x3 matrices. Quaternion input, where used, is explicitly
the OmniGibson xyzw convention. No simulator or model dependency is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vector(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    return array


def _rotation(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {array.shape}")
    if not np.allclose(array.T @ array, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(array), 1.0, atol=1e-6):
        raise ValueError(f"{name} must be a proper rotation matrix")
    return array


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector(self.position, "position"))
        object.__setattr__(self, "rotation", _rotation(self.rotation, "rotation"))


@dataclass(frozen=True)
class PoseDelta:
    translation: np.ndarray
    axis_angle: np.ndarray


def quaternion_xyzw_to_matrix(quaternion: object) -> np.ndarray:
    """Convert a normalized-or-not xyzw quaternion to a 3x3 rotation matrix."""
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quat.shape}")
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def axis_angle_from_rotation_matrix(rotation: object) -> np.ndarray:
    """Return the principal axis-angle vector for a proper rotation matrix."""
    matrix = _rotation(rotation, "rotation")
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if np.pi - angle < 1e-6:
        # Stable axis recovery near pi; choose signs from antisymmetric terms.
        axis = np.sqrt(np.maximum((np.diag(matrix) + 1.0) / 2.0, 0.0))
        axis[0] = np.copysign(axis[0], matrix[2, 1] - matrix[1, 2])
        axis[1] = np.copysign(axis[1], matrix[0, 2] - matrix[2, 0])
        axis[2] = np.copysign(axis[2], matrix[1, 0] - matrix[0, 1])
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            raise ValueError("could not recover axis from pi rotation")
        return axis / norm * angle
    axis = np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    ) / (2.0 * np.sin(angle))
    return axis * angle


def relative_translation_local(current_position: object, current_rotation: object, future_position: object) -> np.ndarray:
    """Compute R_t.T @ (p_future - p_t), exactly as XR-1 _arm_action()."""
    return _rotation(current_rotation, "current_rotation").T @ (
        _vector(future_position, "future_position") - _vector(current_position, "current_position")
    )


def relative_rotation_axis_angle(current_rotation: object, future_rotation: object) -> np.ndarray:
    """Compute axis_angle(R_t.T @ R_future), exactly as XR-1 _arm_action()."""
    return axis_angle_from_rotation_matrix(
        _rotation(current_rotation, "current_rotation").T @ _rotation(future_rotation, "future_rotation")
    )


def relative_pose_delta(current_pose: Pose, future_pose: Pose) -> PoseDelta:
    """Return XR-1 local translation and axis-angle target for a future pose."""
    return PoseDelta(
        translation=relative_translation_local(current_pose.position, current_pose.rotation, future_pose.position),
        axis_angle=relative_rotation_axis_angle(current_pose.rotation, future_pose.rotation),
    )
