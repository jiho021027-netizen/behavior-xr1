"""Explicit, partial R1Pro-proprio to BEHAVIOR-specific XR-1 state packing."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from behavior_xr1.data.schema import BEHAVIOR_R1PRO_PROPRIO_DIM, XR1_STATE_DIM


GripperScalar = Callable[[np.ndarray], float]


def r1pro_gripper_width_sum(finger_qpos: np.ndarray) -> float:
    """Return the two-finger opening-width representation used by 2025's winner.

    This is deliberately an opt-in reducer.  The winning source sums the pair
    and its R1Pro controller is configured in ``smooth`` (symmetric) mode, but
    a 2026 dataset-row check is still required before making it the default.
    """
    pair = np.asarray(finger_qpos, dtype=np.float64)
    if pair.shape != (2,):
        raise ValueError(f"R1Pro gripper qpos must have shape (2,), got {pair.shape}")
    if not np.isfinite(pair).all():
        raise ValueError("R1Pro gripper qpos must be finite")
    return float(pair.sum())


class BehaviorStatePacker:
    """Pack source-confirmed state fields; gripper reduction must be supplied.

    R1Pro exposes two gripper qpos values per hand while XR-1 uses one. This
    class requires an explicit caller-owned reduction so that a mean, a finger
    selection, or a controller-specific mapping is never silently invented.
    """

    def __init__(self, gripper_scalar: GripperScalar | None = None) -> None:
        self._gripper_scalar = gripper_scalar

    def pack(self, state: object) -> np.ndarray:
        source = np.asarray(state, dtype=np.float64)
        if source.ndim < 1 or source.shape[-1] != BEHAVIOR_R1PRO_PROPRIO_DIM:
            raise ValueError(f"BEHAVIOR R1Pro state must have trailing shape ({BEHAVIOR_R1PRO_PROPRIO_DIM},), got {source.shape}")
        if self._gripper_scalar is None:
            raise NotImplementedError("R1Pro 2D gripper qpos requires an explicit gripper_scalar callback")
        flat = source.reshape(-1, BEHAVIOR_R1PRO_PROPRIO_DIM)
        packed = np.zeros((flat.shape[0], XR1_STATE_DIM), dtype=np.float32)
        packed[:, 0:7] = flat[:, 3:10]
        packed[:, 8:15] = flat[:, 28:35]
        for index, row in enumerate(flat):
            packed[index, 7] = self._coerce_scalar(self._gripper_scalar(row[24:26]), "left")
            packed[index, 15] = self._coerce_scalar(self._gripper_scalar(row[49:51]), "right")
        return packed.reshape(source.shape[:-1] + (XR1_STATE_DIM,))

    @staticmethod
    def _coerce_scalar(value: object, arm: str) -> float:
        scalar = np.asarray(value, dtype=np.float64)
        if scalar.shape != ():
            raise ValueError(f"{arm} gripper_scalar callback must return a scalar, got {scalar.shape}")
        if not np.isfinite(scalar):
            raise ValueError(f"{arm} gripper_scalar callback must return a finite value")
        return float(scalar)
