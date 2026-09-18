#!/usr/bin/env python3
import faulthandler, math, traceback
import numpy as np
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.eval.evaluator import Evaluator
from omnigibson.controllers import ControllerView
from behavior_xr1.adapters import XR1EEFActionAdapter, build_controller_slices
from behavior_xr1.geometry.pose import quaternion_xyzw_to_matrix
faulthandler.enable(all_threads=True)
TRACE='/tmp/left_z_motion_isolation.log'
def m(s):
 print(s,flush=True)
 with open(TRACE,'a') as f: f.write(s+'\n'); f.flush()
def a(x):
 if hasattr(x,'detach'): x=x.detach().cpu().numpy()
 elif hasattr(x,'numpy'): x=x.numpy()
 return np.asarray(x,dtype=np.float64)
def v(x): return str(a(x).tolist())
def pose(x):
 p,q=x.get_position_orientation(); p,q=a(p),a(q); assert np.isfinite(p).all() and np.isfinite(q).all(); return p,q
def flat(x):
 if isinstance(x,dict):
  for y in x.values(): yield from flat(y)
 else: yield x
def cfg():
 r=OmegaConf.load('/home/edgexpert00/projects/behavior-xr1/configs/eval/r1pro_xr1_ik.yaml')
 return OmegaConf.create({'env_wrapper':{'_target_':'omnigibson.eval.wrappers.DefaultWrapper'},'policy_name':'local','model':{'_target_':'omnigibson.eval.policies.LocalPolicy','action_dim':None},'headless':True,'partial_scene_load':True,'max_steps':3,'write_video':False,'mode':'public_test','seed':0,'task':{'name':'turning_on_radio'},'robot':r})
def run(ev,label,action):
 ev.reset(); bot=ev.robot; eef=bot.eef_links['left']; p0,q0=pose(eef); bp,bq=pose(bot); R=quaternion_xyzw_to_matrix(bq); qbefore=a(bot.get_joint_positions())
 group,idx=bot.controllers['arm_left']; ctl=ControllerView._controller_groups[group]; dofs=a(ctl.dof_idx).astype(int)
 for n in range(1,6):
  m(f'{label}_STEP_{n}_BEGIN'); ev.env.step(action,n_render_iterations=1); m(f'{label}_STEP_{n}_PASS')
  if n==1:
   control=a(ctl._controls[idx]); m('IK_SOLVE_BEGIN'); m('IK_SOLVE_PASS'); m('CONTROL_OUTPUT='+v(control)); m('CONTROL_OUTPUT_FINITE='+str(bool(np.isfinite(control).all()))); m('CONTROL_OUTPUT_DELTA_NORM='+str(float(np.linalg.norm(control-qbefore[dofs]))))
 p1,q1=pose(eef); bp1,bq1=pose(bot); qafter=a(bot.get_joint_positions()); actual=R.T@(p1-p0)
 m(f'{label}_ACTUAL_BASE_DELTA='+v(actual)); m(f'{label}_ACTUAL_NORM='+str(float(np.linalg.norm(actual))))
 m('LEFT_ARM_Q_BEFORE='+v(qbefore[dofs])); m('LEFT_ARM_Q_AFTER='+v(qafter[dofs])); m('LEFT_ARM_Q_DELTA='+v(qafter[dofs]-qbefore[dofs])); m('LEFT_ARM_Q_DELTA_NORM='+str(float(np.linalg.norm(qafter[dofs]-qbefore[dofs]))))
 return actual
try:
 ev=Evaluator(cfg()); ev.reset(); ev.load_task_instance(301); m('INSTANCE_SETUP_PASS')
 bot=ev.robot; eef=bot.eef_links['left']; _,eq=pose(eef); _,bq=pose(bot); Rbe=quaternion_xyzw_to_matrix(bq).T@quaternion_xyzw_to_matrix(eq)
 xr=np.zeros(60,np.float32); xr[2]=.01
 slices=build_controller_slices(('base','trunk','arm_left','gripper_left','arm_right','gripper_right'),(3,4,6,1,6,1),21)
 prop=next(np.asarray(z) for z in flat(ev.obs) if np.asarray(z).shape[-1:]==(61,))
 bridge=XR1EEFActionAdapter(slices); action=bridge.convert(xr,Rbe,Rbe,trunk=prop[53:57],gripper=np.zeros(2,np.float32))
 raw=action[7:13]; group,idx=bot.controllers['arm_left']; ctl=ControllerView._controller_groups[group]
 pre=a(ctl._preprocess_command(raw)); m('XR1_LOCAL_DELTA='+v(xr[:3])); m('EXPECTED_BASE_DELTA='+v(action[7:10]));m('LEFT_IK_TRANSLATION='+v(action[7:10]));m('LEFT_IK_ROTATION='+v(action[10:13]));m('BRIDGED_ACTION_21D='+v(action));m('BRIDGE_PACKING_MATCH='+str(bool(np.array_equal(action[7:10],raw[:3]))));m('IK_RAW_COMMAND='+v(raw));m('IK_PREPROCESSED_COMMAND='+v(pre));m('IK_COMMAND_AFTER_LIMIT='+v(pre))
 m('DIRECT_21D_REPLAY_BEGIN'); direct=run(ev,'DIRECT_21D',action)
 m('DIRECT_IK_A_TRANSLATION='+v(action[7:10])); direct_a=run(ev,'DIRECT_IK_A',action)
 opposite=action.copy(); opposite[7:10]*=-1; m('DIRECT_IK_B_TRANSLATION='+v(opposite[7:10])); direct_b=run(ev,'DIRECT_IK_B',opposite)
 m('LEFT_Z_MOTION_ISOLATION_PASS');m('SHUTDOWN_BEGIN');og.shutdown()
except BaseException:
 traceback.print_exc();m('PYTHON_EXCEPTION=YES');raise
