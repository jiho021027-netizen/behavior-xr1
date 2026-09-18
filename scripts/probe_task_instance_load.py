#!/usr/bin/env python3
"""Diagnostic-only mirror of BEHAVIOR-1K v3.9.2 Evaluator.load_task_instance."""
import json, os
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.eval.evaluator import Evaluator
from omnigibson.eval.utils.eval_utils import generate_basic_environment_config
from omnigibson.eval.utils import eval_utils
from omnigibson.eval.evaluator import resolve_instance_ids
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.bddl_utils import is_system_bddl_inst
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.eval.utils.light_utils import set_light_control_toggles
TRACE=os.environ.get('TRACE_LOG','/tmp/task_instance_probe.log')
def m(s):
 print(s,flush=True)
 with open(TRACE,'a') as f:f.write(s+'\n');f.flush();os.fsync(f.fileno())
# Build the same official eval cfg; Evaluator owns the actual environment construction.
robot=OmegaConf.load('/home/edgexpert00/projects/behavior-xr1/configs/eval/r1pro_xr1_ik.yaml')
cfg=OmegaConf.create({'env_wrapper':{'_target_':'omnigibson.eval.wrappers.DefaultWrapper'},'policy_name':'local','model':{'_target_':'omnigibson.eval.policies.LocalPolicy','action_dim':None},'headless':True,'partial_scene_load':True,'max_steps':3,'write_video':False,'mode':'public_test','seed':0,'task':{'name':'turning_on_radio'},'robot':robot})
m('INSTANCE_00_BEGIN'); ev=Evaluator(cfg); ev.reset(); m('EVAL_RESET_1_PASS')
instance_id=301; task=ev.env.task; scene_model=task.scene_name
m('INSTANCE_01_FILENAME_BEGIN'); m('SCENE_MODEL='+str(scene_model));m('ACTIVITY_NAME='+str(task.activity_name));m('ACTIVITY_DEFINITION_ID='+str(task.activity_definition_id));m('INSTANCE_ID=301')
tro_filename=task.get_cached_activity_scene_filename(scene_model=scene_model,activity_name=task.activity_name,activity_definition_id=task.activity_definition_id,activity_instance_id=instance_id);m('TRO_FILENAME='+str(tro_filename));m('INSTANCE_02_FILENAME_PASS')
m('INSTANCE_03_PATH_BEGIN'); tro_file_path=get_task_instance_path(scene_model,f'{scene_model}_task_{task.activity_name}_instances/{tro_filename}-tro_state',mode='public_test');m('TRO_FILE_PATH='+str(tro_file_path));m('TRO_FILE_EXISTS='+str(os.path.isfile(tro_file_path)));m('TRO_FILE_SIZE='+str(os.path.getsize(tro_file_path)));m('INSTANCE_04_PATH_PASS')
m('INSTANCE_05_JSON_LOAD_BEGIN');
with open(tro_file_path) as f: raw=json.load(f)
m('TRO_TOP_LEVEL_KEYS='+','.join(raw.keys()));m('INSTANCE_06_JSON_LOAD_PASS');m('INSTANCE_07_TORCH_CONVERT_BEGIN');tro=recursively_convert_to_torch(raw);m('INSTANCE_08_TORCH_CONVERT_PASS')
for i,(key,state) in enumerate(tro.items()):
 m(f'TRO_KEY_BEGIN={key}');m('TRO_KEY_INDEX='+str(i));m('TRO_KEY_NAME='+str(key))
 if key=='robot_poses':
  m('ROBOT_POSES_BEGIN');poses={k.lower():v for k,v in state.items()}; source='robot' if 'robot' in poses else ev.robot.model;m('ROBOT_POSE_SOURCE='+source);m('ROBOT_SET_INSTANCE_POSE_BEGIN');ev.robot.set_position_orientation(poses[source][0]['position'],poses[source][0]['orientation']);m('ROBOT_SET_INSTANCE_POSE_PASS');m('WRITE_ROBOT_METADATA_BEGIN');ev.env.scene.write_task_metadata(key=key,data=state);m('WRITE_ROBOT_METADATA_PASS');m('ROBOT_POSES_PASS')
 else:
  ent=task.object_scope[key];m('OBJECT_STATE_BEGIN='+key);m('OBJECT_SCOPE_ENTITY_TYPE='+str(type(ent)));m('OBJECT_SCOPE_ENTITY_CLASS='+type(ent).__name__);m('OBJECT_LOAD_STATE_BEGIN='+key);ent.load_state(state,serialized=False);m('OBJECT_LOAD_STATE_PASS='+key)
 m('TRO_KEY_PASS='+key)
m('ALL_TRO_STATE_LOAD_PASS');m('UPDATE_HANDLES_BEGIN');og.sim.update_handles();m('UPDATE_HANDLES_PASS')
for n in range(25):
 m(f'SETTLE_STEP_{n:02d}_BEGIN');og.sim.step_physics();m(f'SETTLE_STEP_{n:02d}_PHYSICS_PASS')
 for key,ent in task.object_scope.items():
  if not is_system_bddl_inst(key) and ent is not None:m(f'KEEP_STILL_BEGIN step={n} key={key} class={type(ent).__name__}');ent.keep_still();m('KEEP_STILL_PASS')
 m(f'SETTLE_STEP_{n:02d}_KEEP_STILL_PASS')
m('UPDATE_INITIAL_FILE_BEGIN');ev.env.scene.update_initial_file();m('UPDATE_INITIAL_FILE_PASS');m('SCENE_RESET_BEGIN');ev.env.scene.reset();m('SCENE_RESET_PASS');m('LIGHT_SYNC_RESET_BEGIN');ev._reset_light_synchronizer();m('LIGHT_SYNC_RESET_PASS');m('INSTANCE_LOAD_PASS');m('DIAGNOSTIC_SHUTDOWN_BEGIN');og.shutdown()
