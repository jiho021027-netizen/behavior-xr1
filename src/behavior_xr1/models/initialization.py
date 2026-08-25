"""NumPy-testable contract for pretrained-safe BEHAVIOR dimension activation.

This module does not load a checkpoint or import torch.  The integration code
on the A100 host must apply these operations *immediately after* a successful
XR-1 checkpoint load and before an optimizer is created.
"""

from __future__ import annotations

from collections.abc import MutableMapping

import numpy as np


# BEHAVIOR state trunk qpos (16:20) and local base qvel (20:23) occupy XR-1
# state-tail slots that ``compose_state`` left zero.  The 2026 XR-1 source
# confirms action 17:20 is an existing base part, so it is intentionally not
# included below.  Only trunk action deltas are newly activated.
NEW_STATE_DIMS = np.arange(16, 23, dtype=np.intp)
NEW_ACTION_DIMS = np.arange(20, 24, dtype=np.intp)
XR1_ACTION_DIM = 60
XR1_NUM_CHOICES = 5


def choice_output_rows() -> np.ndarray:
    """Rows for each 60D choice head that correspond to new action dimensions."""
    return np.concatenate([NEW_ACTION_DIMS + choice * XR1_ACTION_DIM for choice in range(XR1_NUM_CHOICES)])


def zero_new_behavior_parameters(parameters: MutableMapping[str, np.ndarray]) -> None:
    """Zero only checkpoint tensors proven to expose new BEHAVIOR dimensions.

    ``parameters`` uses ``xr1.named_parameters()`` names and NumPy arrays in
    the same orientation as PyTorch tensors.  Shape checks guard against a
    source revision changing the architecture.
    """
    expected = {
        "state_projector.layers.0.weight": ("column", NEW_STATE_DIMS, 60),
        "state_projector_choice.layers.0.weight": ("column", NEW_STATE_DIMS, 60),
        "action_projector.layers.0.weight": ("column", NEW_ACTION_DIMS, 60),
        "action_output_layer.layers.2.weight": ("row", NEW_ACTION_DIMS, 60),
        "action_projector_choice.1.layers.0.weight": ("row", choice_output_rows(), XR1_ACTION_DIM * XR1_NUM_CHOICES),
    }
    for name, (axis, indices, required_size) in expected.items():
        if name not in parameters:
            raise KeyError(f"missing XR-1 parameter {name}")
        value = parameters[name]
        if value.ndim != 2 or value.shape[1 if axis == "column" else 0] != required_size:
            raise ValueError(f"unexpected shape for {name}: {value.shape}")
        if axis == "column":
            value[:, indices] = 0
        else:
            value[indices, :] = 0
