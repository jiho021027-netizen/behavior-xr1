#!/usr/bin/env python3
"""Diagnostic-only: isolate the existing live probe's Evaluator.reset() boundary."""
import faulthandler
import traceback

import numpy as np
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.eval.evaluator import Evaluator

from behavior_xr1.geometry.pose import quaternion_xyzw_to_matrix

faulthandler.enable(all_threads=True)
TRACE = "/tmp/sample_reset_isolation.log"


def marker(value: str) -> None:
    print(value, flush=True)
    with open(TRACE, "a", encoding="utf-8") as stream:
        stream.write(value + "\n")
        stream.flush()


def pose_is_valid(position, quaternion) -> None:
    position = np.asarray(position.numpy())
    quaternion = np.asarray(quaternion.numpy())
    assert position.shape == (3,)
    assert quaternion.shape == (4,)
    assert np.isfinite(position).all()
    assert np.isfinite(quaternion).all()


def read_pose(entity, prefix: str):
    marker(f"{prefix}_POSE_READ_BEGIN")
    position, quaternion = entity.get_position_orientation()
    pose_is_valid(position, quaternion)
    marker(f"{prefix}_POSE_READ_PASS")
    return position, quaternion


try:
    robot_cfg = OmegaConf.load(
        "/home/edgexpert00/projects/behavior-xr1/configs/eval/r1pro_xr1_ik.yaml"
    )
    config = OmegaConf.create(
        {
            "env_wrapper": {"_target_": "omnigibson.eval.wrappers.DefaultWrapper"},
            "policy_name": "local",
            "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
            "headless": True,
            "partial_scene_load": True,
            "max_steps": 3,
            "write_video": False,
            "mode": "public_test",
            "seed": 0,
            "task": {"name": "turning_on_radio"},
            "robot": robot_cfg,
        }
    )
    evaluator = Evaluator(config)
    evaluator.reset()
    evaluator.load_task_instance(301)
    marker("INSTANCE_SETUP_PASS")

    marker("RESET_ISOLATION_BEGIN")
    robot = evaluator.robot
    base_position, base_quaternion = read_pose(robot, "PRE_RESET_BASE")
    left_eef = robot.eef_links["left"]
    left_position, left_quaternion = read_pose(left_eef, "PRE_RESET_LEFT_EEF")
    right_eef = robot.eef_links["right"]
    right_position, right_quaternion = read_pose(right_eef, "PRE_RESET_RIGHT_EEF")

    # Exact sample-boundary sequence from /tmp/live_eef_translation.py.
    marker("SAMPLE_RESET_01_BEGIN=Evaluator.reset")
    evaluator.reset()
    marker("SAMPLE_RESET_01_PASS")
    marker("SAMPLE_RESET_PASS")

    marker("POST_RESET_ROBOT_LOOKUP_BEGIN")
    robot = evaluator.robot
    marker("POST_RESET_ROBOT_LOOKUP_PASS")
    marker("ROBOT_REFERENCE_REACQUIRED=YES")

    base_position, base_quaternion = read_pose(robot, "POST_RESET_BASE")

    marker("POST_RESET_LEFT_EEF_LOOKUP_BEGIN")
    left_eef = robot.eef_links["left"]
    marker("POST_RESET_LEFT_EEF_LOOKUP_PASS")
    marker("LEFT_EEF_REFERENCE_REACQUIRED=YES")
    left_position, left_quaternion = read_pose(left_eef, "POST_RESET_LEFT_EEF")

    marker("POST_RESET_RIGHT_EEF_LOOKUP_BEGIN")
    right_eef = robot.eef_links["right"]
    marker("POST_RESET_RIGHT_EEF_LOOKUP_PASS")
    marker("RIGHT_EEF_REFERENCE_REACQUIRED=YES")
    right_position, right_quaternion = read_pose(right_eef, "POST_RESET_RIGHT_EEF")

    for label, quaternion in (
        ("BASE", base_quaternion),
        ("LEFT", left_quaternion),
        ("RIGHT", right_quaternion),
    ):
        marker(f"POST_RESET_{label}_QUAT2MAT_BEGIN")
        rotation = quaternion_xyzw_to_matrix(quaternion.numpy())
        assert rotation.shape == (3, 3)
        assert np.isfinite(rotation).all()
        marker(f"POST_RESET_{label}_QUAT2MAT_PASS")

    marker("POST_RESET_PRESTATE_PASS")
    marker("SHUTDOWN_BEGIN")
    og.shutdown()
except BaseException:
    traceback.print_exc()
    marker("PYTHON_EXCEPTION=YES")
    raise
