from __future__ import annotations
from typing import Any, Callable
import numpy as np
class XR1PolicyWrapper:
    """Thin composition boundary; real upstream loading is explicit and lazy."""
    def __init__(self, backend: Callable[[Any], Any] | None = None): self.backend=backend
    def __call__(self, observation: Any) -> np.ndarray:
        if self.backend is None: raise RuntimeError('XR-1 backend/checkpoint is not configured; refusing implicit model download')
        out=np.asarray(self.backend(observation));
        if out.shape[-1] != 60: raise ValueError(f'XR-1 output must end in 60, got {out.shape}')
        if not np.isfinite(out).all(): raise ValueError('XR-1 output contains non-finite values')
        return out
