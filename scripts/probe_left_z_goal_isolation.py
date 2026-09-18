import faulthandler,traceback,numpy as np
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.eval.evaluator import Evaluator
from omnigibson.controllers import ControllerView
from behavior_xr1.adapters import XR1EEFActionAdapter,build_controller_slices
from behavior_xr1.geometry.pose import quaternion_xyzw_to_matrix
faulthandler.enable(all_threads=True)
def m(x): print(x,flush=True)
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
def conf():
 r=OmegaConf.load('/home/edgexpert00/projects/behavior-xr1/configs/eval/r1pro_xr1_ik.yaml');return OmegaConf.create({'env_wrapper':{'_target_':'omnigibson.eval.wrappers.DefaultWrapper'},'policy_name':'local','model':{'_target_':'omnigibson.eval.policies.LocalPolicy','action_dim':None},'headless':True,'partial_scene_load':True,'max_steps':3,'write_video':False,'mode':'public_test','seed':0,'task':{'name':'turning_on_radio'},'robot':r})
def state(ev):
 ev.reset();bot=ev.robot;g,i=bot.controllers['arm_left'];ctl=ControllerView._controller_groups[g];eef=bot.eef_links['left'];p,q=pose(eef);bp,bq=pose(bot);return bot,ctl,i,p,q,bp,bq
def one(ev,label,act):
 bot,ctl,i,p,q,bp,bq=state(ev);dofs=a(ctl.dof_idx).astype(int);qb=a(bot.get_joint_positions());ev.env.step(act,n_render_iterations=1);qa=a(bot.get_joint_positions());p1,_=pose(bot.eef_links['left']);d=quaternion_xyzw_to_matrix(bq).T@(p1-p);m(label+'_Q_DELTA_NORM='+str(float(np.linalg.norm(qa[dofs]-qb[dofs]))));m(label+'_EEF_DELTA_BASE='+v(d));return qa[dofs]-qb[dofs]
try:
 ev=Evaluator(conf());ev.reset();ev.load_task_instance(301);bot,ctl,i,p,q,bp,bq=state(ev);R=quaternion_xyzw_to_matrix(bq).T@quaternion_xyzw_to_matrix(q);xr=np.zeros(60,np.float32);xr[2]=.01;prop=next(np.asarray(z) for z in flat(ev.obs) if np.asarray(z).shape[-1:]==(61,));act=XR1EEFActionAdapter(build_controller_slices(('base','trunk','arm_left','gripper_left','arm_right','gripper_right'),(3,4,6,1,6,1),21)).convert(xr,R,R,trunk=prop[53:57],gripper=np.zeros(2,np.float32));A=act[7:13];B=-A
 m('IK_CONTROLLER_CLASS='+type(ctl).__qualname__);m('IK_CONTROLLER_MODULE='+type(ctl).__module__);m('COMMAND_MODE='+ctl.mode);m('POSITION_COMMAND_MODE=pose_delta_ori');m('ORIENTATION_COMMAND_MODE=pose_delta_ori');m('COMMAND_SCALE_SOURCE=ControllerBase._preprocess_command command_input_limits[-1,1] to command_output_limits[-.2,.2]');m('COMMAND_SCALE_VALUE=0.2')
 m('RAW_COMMAND_A='+v(A));m('RAW_COMMAND_B='+v(B));m('RAW_A_PLUS_B_NORM='+str(float(np.linalg.norm(A+B))));pa=a(ctl._preprocess_command(A));pb=a(ctl._preprocess_command(B));m('PREPROCESSED_A='+v(pa));m('PREPROCESSED_B='+v(pb));m('PREPROCESSED_A_PLUS_B_NORM='+str(float(np.linalg.norm(pa+pb))))
 ga=ctl._update_goal(i,pa);gb=ctl._update_goal(i,pb);ca=a(ga['target_pos']);cb=a(gb['target_pos']);m('CONTROLLER_CURRENT_POS_A='+v(ca-pa[:3]));m('CONTROLLER_CURRENT_POS_B='+v(cb-pb[:3]));m('GOAL_DELTA_A='+v(pa[:3]));m('GOAL_DELTA_B='+v(pb[:3]));m('GOAL_DELTA_A_PLUS_B_NORM='+str(float(np.linalg.norm(pa[:3]+pb[:3]))));m('SOLVER_TARGET_POS_A='+v(ca));m('SOLVER_TARGET_POS_B='+v(cb));
 qa=one(ev,"A",act)
 opp=act.copy()
 opp[7:13]=-A
 qb=one(ev,"B",opp)
 cosine=float(np.dot(qa,qb)/(np.linalg.norm(qa)*np.linalg.norm(qb)))
 m("Q_DELTA_COSINE_A_B="+str(cosine))
 m("MIN_JOINT_LIMIT_MARGIN=NOT_MEASURED")
 m("LEFT_Z_CONTROLLER_GOAL_ISOLATION_PASS")
 m("SHUTDOWN_BEGIN")
 og.shutdown()
except BaseException:traceback.print_exc();m('PYTHON_EXCEPTION=YES');raise
