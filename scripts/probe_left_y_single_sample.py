#!/usr/bin/env python3
"""One bounded LEFT EEF-local +X ActionBridge/runtime diagnostic."""
import faulthandler
import math
import traceback

import numpy as np
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.eval.evaluator import Evaluator

from behavior_xr1.adapters import XR1EEFActionAdapter, build_controller_slices
from behavior_xr1.geometry.pose import quaternion_xyzw_to_matrix

faulthandler.enable(all_threads=True)
TRACE = "/tmp/left_x_single_sample.log"
DELTA = np.float32(0.01)
STEPS = 5


def marker(value: str) -> None:
    print(value, flush=True)
    with open(TRACE, "a", encoding="utf-8") as stream:
        stream.write(value + "\n")
        stream.flush()


def vector(value) -> np.ndarray:
    result = np.asarray(value.numpy(), dtype=np.float64)
    assert np.isfinite(result).all()
    return result


def pose(entity):
    position, quaternion = entity.get_position_orientation()  # default frame="world"
    position, quaternion = vector(position), vector(quaternion)
    assert position.shape == (3,) and quaternion.shape == (4,)
    return position, quaternion


def flatten(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from flatten(nested)
    else:
        yield value


def format_vector(value) -> str:
    return str(np.asarray(value, dtype=np.float64).tolist())


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

    marker("LEFT_Y_SAMPLE_BEGIN")
    marker("LEFT_Y_PRESTATE_BEGIN")
    robot = evaluator.robot
    left_eef = robot.eef_links["left"]
    left_before, left_quat_before = pose(left_eef)
    base_before, base_quat_before = pose(robot)
    marker("EEF_POSITION_FRAME=world (get_position_orientation default)")
    marker("BASE_POSITION_FRAME=world (get_position_orientation default)")
    marker("LEFT_EEF_POSITION_BEFORE=" + format_vector(left_before))
    marker("LEFT_EEF_QUAT_BEFORE=" + format_vector(left_quat_before))
    marker("BASE_POSITION_BEFORE=" + format_vector(base_before))
    marker("BASE_QUAT_BEFORE=" + format_vector(base_quat_before))
    marker("LEFT_Y_PRESTATE_PASS")

    marker("XR1_ACTION_BUILD_BEGIN")
    xr1_action = np.zeros(60, dtype=np.float32)
    xr1_action[1] = DELTA
    assert xr1_action.shape == (60,)
    assert np.isfinite(xr1_action).all()
    assert np.array_equal(xr1_action[3:6], np.zeros(3, dtype=np.float32))
    assert np.array_equal(xr1_action[8:15], np.zeros(7, dtype=np.float32))
    assert np.array_equal(xr1_action[17:20], np.zeros(3, dtype=np.float32))
    marker("XR1_ACTION_DIM=60")
    marker("XR1_ACTION_FINITE=True")
    marker("XR1_LOCAL_DELTA=" + format_vector(xr1_action[0:3]))
    marker("XR1_ACTION_BUILD_PASS")

    marker("ACTION_BRIDGE_BEGIN")
    slices = build_controller_slices(
        ("base", "trunk", "arm_left", "gripper_left", "arm_right", "gripper_right"),
        (3, 4, 6, 1, 6, 1),
        21,
    )
    bridge = XR1EEFActionAdapter(slices)
    proprioception = next(
        np.asarray(item) for item in flatten(evaluator.obs) if np.asarray(item).shape[-1:] == (61,)
    )
    rotation_world_from_base = quaternion_xyzw_to_matrix(base_quat_before)
    rotation_world_from_eef = quaternion_xyzw_to_matrix(left_quat_before)
    rotation_base_from_eef = rotation_world_from_base.T @ rotation_world_from_eef
    bridged = bridge.convert(
        xr1_action,
        rotation_base_from_eef,
        rotation_base_from_eef,
        trunk=proprioception[53:57],
        gripper=np.zeros(2, dtype=np.float32),
    )
    assert bridged.shape == (21,) and np.isfinite(bridged).all()
    left_translation, left_rotation = bridged[7:10], bridged[10:13]
    right_translation, right_rotation = bridged[14:17], bridged[17:20]
    base_command = bridged[0:3]
    left_gripper, right_gripper = bridged[13:14], bridged[20:21]
    assert np.array_equal(base_command, np.zeros(3, dtype=np.float32))
    assert np.array_equal(left_gripper, np.zeros(1, dtype=np.float32))
    assert np.array_equal(right_gripper, np.zeros(1, dtype=np.float32))
    assert np.array_equal(right_translation, np.zeros(3, dtype=np.float32))
    assert np.array_equal(right_rotation, np.zeros(3, dtype=np.float32))
    assert np.array_equal(left_rotation, np.zeros(3, dtype=np.float32))
    marker("BRIDGED_ACTION_DIM=21")
    marker("BRIDGED_ACTION_FINITE=True")
    marker("LEFT_IK_TRANSLATION=" + format_vector(left_translation))
    marker("LEFT_IK_ROTATION=" + format_vector(left_rotation))
    marker("RIGHT_IK_TRANSLATION=" + format_vector(right_translation))
    marker("RIGHT_IK_ROTATION=" + format_vector(right_rotation))
    marker("BASE_COMMAND=" + format_vector(base_command))
    marker("LEFT_GRIPPER_COMMAND=" + format_vector(left_gripper))
    marker("RIGHT_GRIPPER_COMMAND=" + format_vector(right_gripper))
    expected_base_delta = np.asarray(left_translation, dtype=np.float64)
    marker("EXPECTED_BASE_DELTA=" + format_vector(expected_base_delta))
    marker("EXPECTED_NORM=" + str(float(np.linalg.norm(expected_base_delta))))
    marker("COMMAND_CLIPPED=" + ("YES" if np.any(np.abs(left_translation) > 0.2) else "NO"))
    marker("ACTION_BRIDGE_PASS")

    for step in range(1, STEPS + 1):
        marker(f"LEFT_Y_STEP_{step}_BEGIN")
        evaluator.env.step(bridged, n_render_iterations=1)
        marker(f"LEFT_Y_STEP_{step}_PASS")
        position, _ = pose(left_eef)
        marker(f"LEFT_Y_STEP_{step}_EEF_POS=" + format_vector(position))

    marker("LEFT_Y_POSTSTATE_BEGIN")
    left_after, left_quat_after = pose(left_eef)
    base_after, base_quat_after = pose(robot)
    marker("LEFT_EEF_POSITION_AFTER=" + format_vector(left_after))
    marker("LEFT_EEF_QUAT_AFTER=" + format_vector(left_quat_after))
    marker("BASE_POSITION_AFTER=" + format_vector(base_after))
    marker("BASE_QUAT_AFTER=" + format_vector(base_quat_after))
    marker("LEFT_Y_POSTSTATE_PASS")

    actual_world_delta = left_after - left_before
    actual_base_delta = rotation_world_from_base.T @ actual_world_delta
    base_translation_drift = base_after - base_before
    base_quat_before /= np.linalg.norm(base_quat_before)
    base_quat_after /= np.linalg.norm(base_quat_after)
    base_rotation_drift = 2.0 * math.acos(float(np.clip(abs(np.dot(base_quat_before, base_quat_after)), -1.0, 1.0)))
    actual_norm = float(np.linalg.norm(actual_base_delta))
    expected_norm = float(np.linalg.norm(expected_base_delta))
    marker("ACTUAL_WORLD_DELTA=" + format_vector(actual_world_delta))
    marker("ACTUAL_BASE_DELTA=" + format_vector(actual_base_delta))
    marker("ACTUAL_NORM=" + str(actual_norm))
    marker("BASE_TRANSLATION_DRIFT=" + format_vector(base_translation_drift))
    marker("BASE_ROTATION_DRIFT=" + str(base_rotation_drift))
    base_contamination = np.linalg.norm(base_translation_drift) > 1e-5 or base_rotation_drift > 1e-5
    marker("BASE_CONTAMINATION=" + ("YES" if base_contamination else "NO"))
    if actual_norm <= 1e-8:
        marker("MOTION_DETECTED=NO")
        marker("COSINE_SIMILARITY=UNDEFINED")
        marker("PRIMARY_COMPONENT=UNDEFINED")
        marker("ORTHOGONAL_COMPONENT_NORM=UNDEFINED")
    else:
        cosine = float(np.dot(expected_base_delta, actual_base_delta) / (expected_norm * actual_norm))
        direction = expected_base_delta / expected_norm
        primary = float(np.dot(actual_base_delta, direction))
        orthogonal = float(np.linalg.norm(actual_base_delta - primary * direction))
        marker("MOTION_DETECTED=YES")
        marker("COSINE_SIMILARITY=" + str(cosine))
        marker("PRIMARY_COMPONENT=" + str(primary))
        marker("ORTHOGONAL_COMPONENT_NORM=" + str(orthogonal))
    marker("LEFT_Y_ANALYSIS_PASS")
    marker("SHUTDOWN_BEGIN")
    og.shutdown()
except BaseException:
    traceback.print_exc()
    marker("PYTHON_EXCEPTION=YES")
    raise
