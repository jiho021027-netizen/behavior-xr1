# XR-1 60D state (verified source)

Source: Xiaomi-Robotics-1 `xr1/mibot/utils/io.py::compose_state`, commit `556cca33963a2b36d835a40374c3b4c8eef68401`.

| Slice | Meaning | Evidence |
|---|---|---|
| 0:7 | left arm joint values (up to 7) | `compose_state` |
| 7:8 | left scalar gripper | `compose_state` |
| 8:15 | right arm joint values (up to 7) | `compose_state` |
| 15:16 | right scalar gripper | `compose_state` |
| 16:60 | zero padding/reserved | `np.zeros`, no writes |

No base, EEF, waist, units, or frame are present in XR-1 state packing. Normalization is separate (`normalize_quantile`).
