"""Reference-compatible XR-1 normalization statistics.

This module mirrors the public Xiaomi-Robotics-1 producer at commit
0dd7aef8dc87296246aae812a1f59ccb708e5546.  It deliberately keeps statistics
and runtime tail padding separate: statistics use complete action windows,
while a short runtime window repeats its final target and masks the repeated
steps out.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


ACTION_LENGTH = 30
STATE_DIM = 60
ACTION_DIM = 60
ACTION_EPS = 1e-6


@dataclass(frozen=True)
class OfficialNormalizationStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray


def compute_statistics(states: np.ndarray, complete_actions: np.ndarray) -> OfficialNormalizationStats:
    """Compute the public producer's exact population statistics.

    ``states`` is ``(frames, 1, 60)`` and ``complete_actions`` is
    ``(windows, 30, 60)``.  No padded action rows may be supplied.
    """
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(complete_actions, dtype=np.float32)
    if states.ndim != 3 or states.shape[1:] != (1, STATE_DIM):
        raise ValueError(f"states must have shape (N,1,{STATE_DIM}), got {states.shape}")
    if actions.ndim != 3 or actions.shape[1:] != (ACTION_LENGTH, ACTION_DIM):
        raise ValueError(
            f"complete_actions must have shape (N,{ACTION_LENGTH},{ACTION_DIM}), got {actions.shape}"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("normalization inputs must be finite")
    return OfficialNormalizationStats(
        mean=actions.mean(axis=0),
        std=actions.std(axis=0),
        q01=np.quantile(states, 0.01, axis=0),
        q99=np.quantile(states, 0.99, axis=0),
    )


def pad_runtime(values: np.ndarray, steps: int, length: int = ACTION_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """Repeat the final target for a short runtime chunk and mask repeats out."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim < 1 or not 0 < steps <= length or values.shape[0] != steps:
        raise ValueError(f"values must begin with exactly steps={steps} rows")
    if steps == length:
        padded = values
    else:
        padded = np.concatenate((values, np.repeat(values[-1:], length - steps, axis=0)), axis=0)
    temporal_mask = np.zeros(length, dtype=np.int32)
    temporal_mask[:steps] = 1
    return padded, temporal_mask


def normalize_state(state: np.ndarray, stats: OfficialNormalizationStats) -> np.ndarray:
    value = np.asarray(state, dtype=np.float32)
    valid = stats.q99 > stats.q01
    out = np.zeros_like(value)
    adjusted_q99 = stats.q99[valid] + ACTION_EPS
    out[valid] = (value[valid] - stats.q01[valid]) / (adjusted_q99 - stats.q01[valid])
    return np.clip(2.0 * out - 1.0, -1.0, 1.0)


def normalize_action(action: np.ndarray, stats: OfficialNormalizationStats) -> np.ndarray:
    return (np.asarray(action, dtype=np.float32) - stats.mean) / (stats.std + ACTION_EPS)
