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
 def cap(d):
  e.reset();b=e.robot;g,i=b.controllers['arm_left'];ctl=ControllerView._controller_groups[g];p,q=pose(b.eef_links['left']);bp,bq=pose(b);qi=a(b.get_joint_positions())[a(ctl.dof_idx).astype(int)];R=quaternion_xyzw_to_matrix(bq).T@quaternion_xyzw_to_matrix(q);prop=next(np.asarray(z) for z in flat(e.obs) if np.asarray(z).shape[-1:]==(61,));x=np.zeros(60,np.float32);x[:3]=d;act=XR1EEFActionAdapter(build_controller_slices(('base','trunk','arm_left','gripper_left','arm_right','gripper_right'),(3,4,6,1,6,1),21)).convert(x,R,R,trunk=prop[53:57],gripper=np.zeros(2,np.float32));pre=ctl._preprocess_command(act[7:13]);goal=ctl._update_goal(i,pre);t=a(ctl.compute_control({'target_pos':goal['target_pos'][None,:],'target_ori_mat':goal['target_ori_mat'][None,:,:]}))[0];return qi,bp,bq,p,q,t
 S=[];z=[]
 for i in range(5):
  Z=cap((0,0,.01));C=cap((0,0,0));S.append(Z[:5]);z.append(np.linalg.norm(Z[5]-C[5]));m(f'Z_RESPONSE_NORM_{i}={z[-1]}')
 x=[]
 for i in range(3):x.append(np.linalg.norm(cap((.01,0,0))[5]-cap((0,0,0))[5]));m(f'X_RESPONSE_NORM_{i}={x[-1]}')
 ref=S[0];mx=[max(np.linalg.norm(s[j]-ref[j]) for s in S) for j in range(5)];
 for k,v in zip(['MAX_Q_RESET_DIFF','MAX_BASE_POS_RESET_DIFF','MAX_BASE_QUAT_RESET_DIFF','MAX_EEF_POS_RESET_DIFF','MAX_EEF_QUAT_RESET_DIFF'],mx):m(k+'='+str(v))
 for k,v in [('MIN_Z_RESPONSE_NORM',min(z)),('MAX_Z_RESPONSE_NORM',max(z)),('MEAN_Z_RESPONSE_NORM',np.mean(z)),('STD_Z_RESPONSE_NORM',np.std(z)),('MEAN_X_RESPONSE_NORM',np.mean(x)),('STD_X_RESPONSE_NORM',np.std(x))]:m(k+'='+str(v))
 # old-style one reset, direct sequential production calls
 e.reset();b=e.robot;g,i=b.controllers['arm_left'];ctl=ControllerView._controller_groups[g];p,q=pose(b.eef_links['left']);qb=a(b.get_joint_positions());R=quaternion_xyzw_to_matrix(pose(b)[1]).T@quaternion_xyzw_to_matrix(q);prop=next(np.asarray(z0) for z0 in flat(e.obs) if np.asarray(z0).shape[-1:]==(61,));br=XR1EEFActionAdapter(build_controller_slices(('base','trunk','arm_left','gripper_left','arm_right','gripper_right'),(3,4,6,1,6,1),21));T={}
 for n,d in [('X',(.01,0,0)),('Y',(0,.01,0)),('Z',(0,0,.01)),('C',(0,0,0))]:
  xx=np.zeros(60,np.float32);xx[:3]=d;ac=br.convert(xx,R,R,trunk=prop[53:57],gripper=np.zeros(2,np.float32));go=ctl._update_goal(i,ctl._preprocess_command(ac[7:13]));T[n]=a(ctl.compute_control({'target_pos':go['target_pos'][None,:],'target_ori_mat':go['target_ori_mat'][None,:,:]}))[0]
 for n in 'XYZ':m('OLD_STYLE_'+n+'_RESPONSE_NORM='+str(np.linalg.norm(T[n]-T['C'])))
 m('OLD_STYLE_Q_CHANGE_NORM='+str(np.linalg.norm(a(b.get_joint_positions())-qb)));m('OLD_STYLE_EEF_CHANGE_NORM='+str(np.linalg.norm(pose(b.eef_links['left'])[0]-p)));m('NEW_STYLE_NO_STEP_X_RESPONSE_NORM='+str(x[0]));m('NEW_STYLE_NO_STEP_Z_RESPONSE_NORM='+str(z[0]));m('RESET_RESULT_MATCHES=NEITHER');m('RESET_REPRODUCIBLE='+str(all(v==0 for v in mx)));m('LEFT_Z_ROOT_CAUSE=UNRESOLVED_DIAGNOSTIC_INCONSISTENCY');m('NEXT_STEP=TRACE_REVIEW');m('SHUTDOWN_BEGIN');og.shutdown()
except BaseException:traceback.print_exc();m('PYTHON_EXCEPTION=YES');raise
