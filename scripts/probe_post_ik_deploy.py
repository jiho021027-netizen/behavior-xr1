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
def v(x):return str(a(x).tolist())
try:
 r=OmegaConf.load('/home/edgexpert00/projects/behavior-xr1/configs/eval/r1pro_xr1_ik.yaml');c=OmegaConf.create({'env_wrapper':{'_target_':'omnigibson.eval.wrappers.DefaultWrapper'},'policy_name':'local','model':{'_target_':'omnigibson.eval.policies.LocalPolicy','action_dim':None},'headless':True,'partial_scene_load':True,'max_steps':3,'write_video':False,'mode':'public_test','seed':0,'task':{'name':'turning_on_radio'},'robot':r});e=Evaluator(c);e.reset();e.load_task_instance(301)
 def run(n,d):
  e.reset();b=e.robot;g,i=b.controllers['arm_left'];ctl=ControllerView._controller_groups[g];eef=b.eef_links['left'];p,q=pose(eef);_,bq=pose(b);R=quaternion_xyzw_to_matrix(bq).T@quaternion_xyzw_to_matrix(q);prop=next(np.asarray(z) for z in flat(e.obs) if np.asarray(z).shape[-1:]==(61,));x=np.zeros(60,np.float32);x[:3]=d;act=XR1EEFActionAdapter(build_controller_slices(('base','trunk','arm_left','gripper_left','arm_right','gripper_right'),(3,4,6,1,6,1),21)).convert(x,R,R,trunk=prop[53:57],gripper=np.zeros(2,np.float32));pre=ctl._preprocess_command(act[7:13]);goal=ctl._update_goal(i,pre);target=a(ctl.compute_control({'target_pos':goal['target_pos'][None,:],'target_ori_mat':goal['target_ori_mat'][None,:,:]}))[0];qb=a(b.get_joint_positions())[a(ctl.dof_idx).astype(int)];e.env.step(act,n_render_iterations=1);dep=a(ctl._controls[i]);qa=a(b.get_joint_positions())[a(ctl.dof_idx).astype(int)];p1,_=pose(eef);return target,dep,qa-qb,R.T@(p1-p)
 X=run('X',(0.01,0,0));Z=run('Z',(0,0,0.01));C=run('C',(0,0,0));
 def norm(x):return float(np.linalg.norm(x))
 for n,t in zip('XZC',(X,Z,C)):m('TARGET_'+n+'='+v(t[0]));m('FINAL_DEPLOYED_TARGET_'+n+'='+v(t[1]))
 tx,tz=X[0]-C[0],Z[0]-C[0];dx,dz=X[1]-C[1],Z[1]-C[1];qx,qz=X[2]-C[2],Z[2]-C[2];ex,ez=X[3]-C[3],Z[3]-C[3]
 for k,x in [('TARGET_RESPONSE_X_NORM',tx),('TARGET_RESPONSE_Z_NORM',tz),('DEPLOYED_RESPONSE_X_NORM',dx),('DEPLOYED_RESPONSE_Z_NORM',dz),('IK_TO_DEPLOY_DIFF_X_NORM',X[1]-X[0]),('IK_TO_DEPLOY_DIFF_Z_NORM',Z[1]-Z[0]),('Q_RESPONSE_X_NORM',qx),('Q_RESPONSE_Z_NORM',qz),('EEF_RESPONSE_X_NORM',ex),('EEF_RESPONSE_Z_NORM',ez)]:m(k+'='+str(norm(x)))
 for k,n,d in [('TARGET_Z_TO_X_RATIO',tz,tx),('DEPLOYED_Z_TO_X_RATIO',dz,dx),('Q_Z_TO_X_RATIO',qz,qx),('EEF_Z_TO_X_RATIO',ez,ex)]:m(k+'='+str(norm(n)/norm(d)))
 m('TARGET_X_AT_LIMIT=NOT_MEASURED');m('TARGET_Z_AT_LIMIT=NOT_MEASURED');m('NEXT_STEP=TRACE_REVIEW');m('SHUTDOWN_BEGIN');og.shutdown()
except BaseException:traceback.print_exc();m('PYTHON_EXCEPTION=YES');raise
