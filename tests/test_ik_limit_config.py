from pathlib import Path
import yaml


def test_custom_ik_limits_are_command_dimensional():
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/eval/r1pro_xr1_ik.yaml").read_text()
    )
    for arm in ("arm_left", "arm_right"):
        controller = cfg["controller_config"][arm]
        assert controller["name"] == "InverseKinematicsController"
        assert controller["mode"] == "pose_delta_ori"
        assert len(controller["command_output_limits"][0]) == 6
        assert len(controller["command_output_limits"][1]) == 6
