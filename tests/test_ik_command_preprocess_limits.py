import torch
import yaml
from pathlib import Path

from omnigibson.controllers.controller_base import BaseController


def _controller(input_limits, output_limits):
    c = object.__new__(BaseController)
    c._command_input_limits = tuple(torch.tensor(x, dtype=torch.float32) for x in input_limits)
    c._command_output_limits = tuple(torch.tensor(x, dtype=torch.float32) for x in output_limits)
    c._command_scale_factor = None
    c._command_output_transform = None
    c._command_input_transform = None
    return c


def test_postfix_ik_preprocess_is_command_dimensional():
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/eval/r1pro_xr1_ik.yaml").read_text())
    for arm in ("arm_left", "arm_right"):
        limits = cfg["controller_config"][arm]["command_output_limits"]
        assert len(limits[0]) == len(limits[1]) == 6
        c = _controller(([-1] * 6, [1] * 6), limits)
        out = c._preprocess_command(torch.zeros(6))
        assert tuple(out.shape) == (6,)
        assert torch.isfinite(out).all()


def test_old_seven_dimensional_limit_reproduces_broadcast_failure():
    c = _controller(([-1] * 6, [1] * 6), ([-1] * 7, [1] * 7))
    try:
        c._preprocess_command(torch.zeros(6))
    except RuntimeError:
        return
    raise AssertionError("7D output limits unexpectedly accepted for a 6D command")
