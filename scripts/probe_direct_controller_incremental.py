#!/usr/bin/env python3
import yaml
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.controllers import ControllerView

gm.HEADLESS = True
cfg = yaml.safe_load(open('/home/edgexpert00/projects/behavior-2026-validation/OmniGibson/omnigibson/configs/r1pro_behavior.yaml'))
for key in ('arm_left', 'arm_right'):
    c = cfg['robots'][0]['controller_config'][key]
    c.clear()
    c.update({'name': 'InverseKinematicsController', 'mode': 'pose_delta_ori',
              'command_input_limits': 'default',
              'command_output_limits': [[-.2, -.2, -.2, -.5, -.5, -.5], [.2, .2, .2, .5, .5, .5]]})
cfg['task']['activity_name'] = 'picking_up_trash'
cfg['task']['activity_definition_id'] = 0
cfg['task']['activity_instance_id'] = 0

print('CTRL_00_BEGIN', flush=True)
env = og.Environment(configs=cfg)
print('CTRL_01_ENV_PASS', flush=True)
robot = env.robots[0]
print('CTRL_02_ROBOT_CLASS', type(robot).__module__, type(robot).__name__, flush=True)
print('CTRL_03_ACTION_DIM', int(robot.action_dim), flush=True)
print('CTRL_04_CONTROLLER_ORDER', list(robot.controller_order), flush=True)
for name in robot.controller_order:
    print('CTRL_BEGIN', name, flush=True)
    group, index = robot.controllers[name]
    controller = ControllerView._controller_groups[group].controllers[index]
    print('CTRL_CLASS', name, type(controller).__module__, type(controller).__name__, flush=True)
    print('CTRL_COMMAND_DIM', name, int(ControllerView.get_command_dim(group)), flush=True)
    for symbol in ('control_dim', 'dof_idx'):
        if hasattr(controller, symbol):
            value = getattr(controller, symbol)
            try: size = len(value)
            except Exception: size = 'NA'
            print('CTRL_PROPERTY', name, symbol, type(value).__name__, size, flush=True)
print('DIRECT_BASIC_SNAPSHOT_PASS', flush=True)
