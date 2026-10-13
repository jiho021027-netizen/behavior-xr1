from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np

@dataclass(frozen=True)
class CanonicalObservation:
    rgb: dict[str, np.ndarray]
    depth: dict[str, np.ndarray]
    proprioception: np.ndarray
    instruction: str | None
    task_id: int | None
    metadata: dict[str, Any]

class RuntimeObservationAdapter:
    """Normalize an official flattened observation without inventing camera names."""
    def __init__(self, *, use_depth: bool = False, rgb_keys: tuple[str,...] = (), depth_keys: tuple[str,...] = (), proprio_key: str = 'proprio'):
        self.use_depth, self.rgb_keys, self.depth_keys, self.proprio_key = use_depth, rgb_keys, depth_keys, proprio_key
    def __call__(self, observation: Mapping[str, Any]) -> CanonicalObservation:
        if not isinstance(observation, Mapping): raise TypeError('observation must be a mapping')
        flat = dict(self._flatten(observation))
        prop = flat.get(self.proprio_key)
        if prop is None: raise KeyError(f'missing proprioception key: {self.proprio_key}')
        prop = np.asarray(prop)
        if prop.shape[-1] != 61: raise ValueError(f'proprioception trailing dimension must be 61, got {prop.shape}')
        rgb = {k: self._rgb(flat[k], k) for k in self.rgb_keys if k in flat}
        missing = set(self.rgb_keys) - set(rgb)
        if missing: raise KeyError(f'missing RGB keys: {sorted(missing)}')
        depth = {k: np.asarray(flat[k]) for k in self.depth_keys if k in flat}
        if self.use_depth and set(depth) != set(self.depth_keys): raise KeyError(f'missing depth keys: {sorted(set(self.depth_keys)-set(depth))}')
        instruction = flat.get('instruction', flat.get('task_instruction'))
        task_id = flat.get('task_id')
        return CanonicalObservation(rgb, depth if self.use_depth else {}, prop, None if instruction is None else str(instruction), None if task_id is None else int(task_id), {'use_depth': self.use_depth, 'rgb_keys': tuple(rgb), 'depth_keys': tuple(depth) if self.use_depth else ()})
    def _flatten(self, x, prefix=''):
        if isinstance(x, Mapping):
            for k,v in x.items(): yield from self._flatten(v, f'{prefix}.{k}' if prefix else str(k))
        else: yield prefix, x
    @staticmethod
    def _rgb(value, key):
        a=np.asarray(value)
        if a.ndim < 3 or a.shape[-1] not in (3,4): raise ValueError(f'RGB key {key} must end in 3/4 channels, got {a.shape}')
        return a[..., :3]
