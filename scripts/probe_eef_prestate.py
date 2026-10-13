import os,traceback,faulthandler,numpy as np
faulthandler.enable(all_threads=True)
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.eval.evaluator import Evaluator
from behavior_xr1.geometry.pose import quaternion_xyzw_to_matrix
L='/tmp/prestate.log'
def m(x):print(x,flush=True);open(L,'a').write(x+'\n')
try:
 r=OmegaConf.load('/home/edgexpert00/projects/behavior-xr1/configs/eval/r1pro_xr1_ik.yaml');c=OmegaConf.create({'env_wrapper':{'_target_':'omnigibson.eval.wrappers.DefaultWrapper'},'policy_name':'local','model':{'_target_':'omnigibson.eval.policies.LocalPolicy','action_dim':None},'headless':True,'partial_scene_load':True,'max_steps':3,'write_video':False,'mode':'public_test','seed':0,'task':{'name':'turning_on_radio'},'robot':r});e=Evaluator(c);e.reset();e.load_task_instance(301);m('INSTANCE_SETUP_PASS');m('SAMPLE_0_BEGIN');m('EXTRA_SAMPLE_RESET=NO');m('ROBOT_LOOKUP_BEGIN');bot=e.robot;m('ROBOT_LOOKUP_PASS');m('BASE_POSE_READ_BEGIN');bp,bq=bot.get_position_orientation();m('BASE_POSE_READ_PASS');m('LEFT_EEF_LOOKUP_BEGIN');le=bot.eef_links['left'];m('LEFT_EEF_LOOKUP_PASS');m('RIGHT_EEF_LOOKUP_BEGIN');re=bot.eef_links['right'];m('RIGHT_EEF_LOOKUP_PASS');m('LEFT_EEF_POSE_READ_BEGIN');lp,lq=le.get_position_orientation();m('LEFT_EEF_POSE_READ_PASS');m('RIGHT_EEF_POSE_READ_BEGIN');rp,rq=re.get_position_orientation();m('RIGHT_EEF_POSE_READ_PASS');m('BASE_QUAT2MAT_BEGIN');quaternion_xyzw_to_matrix(bq.numpy());m('BASE_QUAT2MAT_PASS');m('LEFT_QUAT2MAT_BEGIN');quaternion_xyzw_to_matrix(lq.numpy());m('LEFT_QUAT2MAT_PASS');m('RIGHT_QUAT2MAT_BEGIN');quaternion_xyzw_to_matrix(rq.numpy());m('RIGHT_QUAT2MAT_PASS');m('PRESTATE_ACQUISITION_PASS');m('SHUTDOWN_BEGIN');og.shutdown()
except BaseException:
 traceback.print_exc();m('PYTHON_EXCEPTION=YES');raise
