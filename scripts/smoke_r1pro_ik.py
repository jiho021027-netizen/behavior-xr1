#!/usr/bin/env python3
"""A100/OmniGibson-only static R1Pro controller inspection; no rollout."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="set OmniGibson headless mode before load")
    args = parser.parse_args()
    # Imports are intentionally below parsing: laptop ``--help`` remains safe.
    try:
        import omnigibson as og
    except ImportError as error:
        parser.error(f"OmniGibson host environment is required: {error}")
    from omnigibson.controllers import ControllerView
    from omnigibson.macros import gm

    gm.HEADLESS = args.headless
    # A bare Scene and empty observations keep this a robot/controller load,
    # not a BEHAVIOR task rollout or dataset access.
    cfg = {
        "scene": {"type": "Scene"},
        "objects": [],
        "robots": [{
            "type": "R1Pro", "name": "r1pro_smoke", "obs_modalities": [],
            "action_normalize": False, "fixed_base": False,
            "controller_config": {
                "base": {"name": "HolonomicBaseJointController", "motor_type": "velocity", "use_impedances": False},
                "trunk": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "use_impedances": False},
                "arm_left": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "use_impedances": False},
                "arm_right": {"name": "JointController", "motor_type": "position", "use_delta_commands": False, "use_impedances": False},
                "gripper_left": {"name": "MultiFingerGripperController", "mode": "smooth"},
                "gripper_right": {"name": "MultiFingerGripperController", "mode": "smooth"},
            },
        }],
    }
    env = og.Environment(configs=cfg)
    try:
        robot = env.robots[0]
        print(f"action_dim: {robot.action_dim}")
        print(f"controller_order: {robot.controller_order}")
        for name in robot.controller_order:
            group_key, controller_index = robot.controllers[name]
            print(f"controller {name}: group={group_key}, index={controller_index}, class={ControllerView._controller_groups[group_key].__class__.__name__}, command_dim={ControllerView.get_command_dim(group_key)}")
        print(f"eef_links: { {name: link.prim_path for name, link in robot.eef_links.items()} }")
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
