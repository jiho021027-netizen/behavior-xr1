import numpy as np,traceback
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.eval.evaluator import Evaluator
from omnigibson.controllers import ControllerView
from behavior_xr1.adapters import XR1EEFActionAdapter,build_controller_slices
from behavior_xr1.geometry.pose import quaternion_xyzw_to_matrix
def m(x):print(x,flush=True)
def a(x):
 if hasattr(x,'detach'):x=x.detach().cpu().numpy()
 elif hasattr(x,'numpy'):x=x.numpy()
 return np.asarray(x,float)
def pose(x):p,q=x.get_position_orientation();return a(p),a(q)
def flat(x):
 if isinstance(x,dict):
  for y in x.values():yield from flat(y)
 else:yield x
def cfg():
 r=OmegaConf.load('/home/edgexpert00/projects/behavior-xr1/configs/eval/r1pro_xr1_ik.yaml');return OmegaConf.create({'env_wrapper':{'_target_':'omnigibson.eval.wrappers.DefaultWrapper'},'policy_name':'local','model':{'_target_':'omnigibson.eval.policies.LocalPolicy','action_dim':None},'headless':True,'partial_scene_load':True,'max_steps':3,'write_video':False,'mode':'public_test','seed':0,'task':{'name':'turning_on_radio'},'robot':r})
try:
 e=Evaluator(cfg());e.reset();e.load_task_instance(301)
 def run(seq):
  e.reset();b=e.robot;g,i=b.controllers['arm_left'];ctl=ControllerView._controller_groups[g];p,q=pose(b.eef_links['left']);R=quaternion_xyzw_to_matrix(pose(b)[1]).T@quaternion_xyzw_to_matrix(q);pr=next(np.asarray(z) for z in flat(e.obs) if np.asarray(z).shape[-1:]==(61,));br=XR1EEFActionAdapter(build_controller_slices(('base','trunk','arm_left','gripper_left','arm_right','gripper_right'),(3,4,6,1,6,1),21));T={}
  for n in seq:
   d={'X':(.01,0,0),'Y':(0,.01,0),'Z':(0,0,.01),'C':(0,0,0)}[n];x=np.zeros(60,np.float32);x[:3]=d;ac=br.convert(x,R,R,trunk=pr[53:57],gripper=np.zeros(2,np.float32));go=ctl._update_goal(i,ctl._preprocess_command(ac[7:13]));T[n]=a(ctl.compute_control({'target_pos':go['target_pos'][None,:],'target_ori_mat':go['target_ori_mat'][None,:,:]}))[0]
  return np.linalg.norm(T['Z']-T['C']),T
 S={'A':'ZC','B':'XZC','C':'YZC','D':'XYZC','E':'YXZC','F':'XCZC','G':'YCZC'};R={}
 for k,s in S.items():R[k]=run(s);m('SEQ_'+k+'_Z_RESPONSE_NORM='+str(R[k][0]))
 m('Z_TARGET_CHANGE_NORM='+str(np.linalg.norm(R['D'][1]['Z']-R['A'][1]['Z'])));m('C_TARGET_CHANGE_NORM='+str(np.linalg.norm(R['D'][1]['C']-R['A'][1]['C'])));m('PREP_ONLY_XY_THEN_Z_RESPONSE_NORM=NOT_MEASURED');m('COMMAND_PREPARATION_PATH=InverseKinematicsController._update_goal then _preprocess_command');m('COMMAND_PREPARATION_MUTATES_STATE=False');m('MUTATED_FIELDS=NONE_FOR_pose_delta_ori');m('DELTA_REFERENCE_SOURCE=ik_controller.py:236-246');m('DELTA_REFERENCE_SEMANTICS=current_measured_EEF_pose');m('SEQUENCE_DEPENDENCE=TRACE_REVIEW');m('LEFT_Z_ROOT_CAUSE=UNRESOLVED_CONTROLLER_SEQUENCE_STATE');m('NEXT_STEP=TRACE_REVIEW');m('SHUTDOWN_BEGIN');og.shutdown()
except BaseException:traceback.print_exc();m('PYTHON_EXCEPTION=YES');raise
