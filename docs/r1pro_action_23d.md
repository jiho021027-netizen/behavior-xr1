# R1Pro official 23D action

Source: BEHAVIOR v3.9.2 `OmniGibson/omnigibson/eval/utils/eval_utils.py::ACTION_QPOS_INDICES` and `eval/r1pro.yaml`.

`0:3` base velocity; `3:7` trunk position; `7:14` left arm joint position; `14:15` left scalar gripper; `15:22` right arm joint position; `22:23` right scalar gripper. The bundled evaluator config uses `action_normalize: false`.
