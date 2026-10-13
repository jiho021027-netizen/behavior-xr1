import faulthandler,numpy as np,traceback
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.eval.evaluator import Evaluator
from omnigibson.controllers import ControllerView
from omnigibson.utils.usd_utils import ControllableObjectViewAPI
from behavior_xr1.adapters import XR1EEFActionAdapter,build_controller_slices
from behavior_xr1.geometry.pose import quaternion_xyzw_to_matrix
faulthandler.enable(all_threads=True)
def m(x):print(x,flush=True)
def a(x):
 if hasattr(x,'detach'):x=x.detach().cpu().numpy()
 elif hasattr(x,'numpy'):x=x.numpy()
 return np.asarray(x,dtype=float)
def v(x):return str(a(x).tolist())
def pose(x):p,q=x.get_position_orientation();return a(p),a(q)
def flat(x):
 if isinstance(x,dict):
  for y in x.values():yield from flat(y)
 else:yield x
try:
 r=OmegaConf.load('/home/edgexpert00/projects/behavior-xr1/configs/eval/r1pro_xr1_ik.yaml');c=OmegaConf.create({'env_wrapper':{'_target_':'omnigibson.eval.wrappers.DefaultWrapper'},'policy_name':'local','model':{'_target_':'omnigibson.eval.policies.LocalPolicy','action_dim':None},'headless':True,'partial_scene_load':True,'max_steps':3,'write_video':False,'mode':'public_test','seed':0,'task':{'name':'turning_on_radio'},'robot':r});e=Evaluator(c);e.reset();e.load_task_instance(301);b=e.robot;g,i=b.controllers['arm_left'];ctl=ControllerView._controller_groups[g];p,q=pose(b.eef_links['left']);bp,bq=pose(b);rows=ctl.view_row_indices;aq=ControllableObjectViewAPI.get_all_joint_positions(ctl.routing_path);qa=aq[rows,:][:,ctl.dof_idx];ja=ControllableObjectViewAPI.get_all_relative_jacobians(ctl.routing_path);ei=ControllableObjectViewAPI.get_link_index(ctl.routing_path,ctl._link_name);off=ja.shape[-1]-aq.shape[-1];di=ctl.dof_idx+off;J=ja[rows][:,ei-1,:,:][:,:,di];m('LEFT_JACOBIAN_SHAPE='+str(tuple(J.shape)));m('LEFT_JACOBIAN_FINITE='+str(bool(np.isfinite(a(J)).all())));m('JACOBIAN_FRAME=ROBOT_BASE_RELATIVE');m('JACOBIAN_ROW_ORDER=NOT_FOUND');
 R=quaternion_xyzw_to_matrix(bq).T@quaternion_xyzw_to_matrix(q);prop=next(np.asarray(z) for z in flat(e.obs) if np.asarray(z).shape[-1:]==(61,));br=XR1EEFActionAdapter(build_controller_slices(('base','trunk','arm_left','gripper_left','arm_right','gripper_right'),(3,4,6,1,6,1),21));ts={};
 for n,d in [('X',(0.01,0,0)),('Y',(0,0.01,0)),('Z',(0,0,0.01)),('C',(0,0,0))]:
  x=np.zeros(60,np.float32);x[:3]=d;act=br.convert(x,R,R,trunk=prop[53:57],gripper=np.zeros(2,np.float32));pre=ctl._preprocess_command(act[7:13]);goal=ctl._update_goal(i,pre);out=ctl.compute_control({'target_pos':goal['target_pos'][None,:],'target_ori_mat':goal['target_ori_mat'][None,:,:]});ts[n]=a(out)[0];m('TARGET_'+n+'='+v(ts[n]))
 for n in 'XYZ':m('SOLVER_RESPONSE_'+n+'_NORM='+str(float(np.linalg.norm(ts[n]-ts['C']))))
 m('ZERO_TARGET_DELTA_NORM='+str(float(np.linalg.norm(ts['C']-a(qa)[0]))));m('SHUTDOWN_BEGIN');og.shutdown()
except BaseException:traceback.print_exc();m('PYTHON_EXCEPTION=YES');raise
