"""Interfaces that prohibit shape-only action conversion."""

from __future__ import annotations

from typing import Any

from .state import BehaviorStatePacker


def _length(value: Any, label: str) -> int:
    try:
        return len(value)
    except TypeError as error:
        raise ValueError(f"{label} must be a sized sequence") from error


class BehaviorObservationAdapter:
    """Future RGB/depth/key-selection adapter; no policy is chosen yet."""

    def __call__(self, observation: Any) -> Any:
        raise NotImplementedError("Observation semantics require a confirmed evaluator key contract")


class XR1ActionToBehaviorAction:
    """Future XR-1 EE-space to R1Pro controller converter; intentionally unavailable."""

    def convert(self, action: Any) -> Any:
        if _length(action, "XR-1 action horizon") != 30:
            raise ValueError("XR-1 action must have its audited 30-step horizon")
        for index, step in enumerate(action):
            if _length(step, f"XR-1 action step {index}") != 60:
                raise ValueError(f"XR-1 action step {index} must have 60 values")
        raise NotImplementedError(
            "XR-1 EE-space actions require validated R1Pro controller/kinematic semantics; no slice conversion exists"
        )
