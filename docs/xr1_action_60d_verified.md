# XR-1 60D action (verified source)

Source: `xr1/mibot/utils/io.py::ACTION_PARTS`, `compose_action`, `split_action`, `recover_action`, same upstream commit.

| Slice | Meaning |
|---|---|
| 0:3 | left EEF translation delta |
| 3:6 | left EEF axis-angle delta |
| 6:7 | left gripper delta |
| 8:11 | right EEF translation delta |
| 11:14 | right EEF axis-angle delta |
| 14:15 | right gripper delta |
| 16:17 | waist delta |
| 17:20 | base velocity |
| 7,15,20:60 | reserved zero/masked |

`recover_action` rotates translation by `rotm.T` (EEF-local delta to world/base pose) and composes rotation as `rotm @ aa2rotm(delta)`. Gripper and waist are deltas; base is passed through. This proves direct 60D→23D joint slicing is invalid.
