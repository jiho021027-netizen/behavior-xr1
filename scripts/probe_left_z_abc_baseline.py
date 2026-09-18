import faulthandler,traceback,numpy as np
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.eval.evaluator import Evaluator
from omnigibson.controllers import ControllerView
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
def cfg():
 r=OmegaConf.load('/home/edgexpert00/projects/behavior-xr1/configs/eval/r1pro_xr1_ik.yaml');return OmegaConf.create({'env_wrapper':{'_target_':'omnigibson.eval.wrappers.DefaultWrapper'},'policy_name':'local','model':{'_target_':'omnigibson.eval.policies.LocalPolicy','action_dim':None},'headless':True,'partial_scene_load':True,'max_steps':3,'write_video':False,'mode':'public_test','seed':0,'task':{'name':'turning_on_radio'},'robot':r})
def one(ev,label,action):
 ev.reset();bot=ev.robot;g,i=bot.controllers['arm_left'];ctl=ControllerView._controller_groups[g];d=a(ctl.dof_idx).astype(int);eef=bot.eef_links['left'];p,_=pose(eef);_,bq=pose(bot);q=a(bot.get_joint_positions())[d];m('EEF_POS_INITIAL_'+label+'='+v(p));m('ARM_Q_INITIAL_'+label+'='+v(q));ev.env.step(action,n_render_iterations=1);out=a(ctl._controls[i]);qa=a(bot.get_joint_positions())[d];p1,_=pose(eef);ed=quaternion_xyzw_to_matrix(bq).T@(p1-p);return out,q,qa-q,ed
try:
 ev=Evaluator(cfg());ev.reset();ev.load_task_instance(301);bot=ev.robot;_,eq=pose(bot.eef_links['left']);_,bq=pose(bot);R=quaternion_xyzw_to_matrix(bq).T@quaternion_xyzw_to_matrix(eq);xr=np.zeros(60,np.float32);xr[2]=.01;prop=next(np.asarray(z) for z in flat(ev.obs) if np.asarray(z).shape[-1:]==(61,));base=XR1EEFActionAdapter(build_controller_slices(('base','trunk','arm_left','gripper_left','arm_right','gripper_right'),(3,4,6,1,6,1),21)).convert(xr,R,R,trunk=prop[53:57],gripper=np.zeros(2,np.float32));A=base.copy();B=base.copy();B[7:13]*=-1;C=base.copy();C[7:13]=0
 oa,qi,qa,ea=one(ev,'A',A);ob,qj,qb,eb=one(ev,'B',B);oc,qk,qc,ec=one(ev,'C',C)
 for n,x in [('A',oa),('B',ob),('C',oc)]:m('CONTROL_OUTPUT_'+n+'='+v(x));m('CONTROL_DELTA_'+n+'='+v(x-{'A':qi,'B':qj,'C':qk}[n]));m('CONTROL_DELTA_'+n+'_NORM='+str(float(np.linalg.norm(x-{'A':qi,'B':qj,'C':qk}[n]))))
 da,db,dc=oa-qi,ob-qj,oc-qk;m('CONTROL_A_B_DIFF_NORM='+str(float(np.linalg.norm(da-db))));m('CONTROL_A_MINUS_C_NORM='+str(float(np.linalg.norm(da-dc))));m('CONTROL_B_MINUS_C_NORM='+str(float(np.linalg.norm(db-dc))));m('CONTROL_DELTA_COSINE_A_B='+str(float(np.dot(da,db)/(np.linalg.norm(da)*np.linalg.norm(db)))));m('CONTROL_OUTPUTS_NEAR_IDENTICAL='+str(bool(np.allclose(da,db))))
 for n,x in [('A',qa),('B',qb),('C',qc)]:m('Q_DELTA_'+n+'='+v(x));m('Q_DELTA_'+n+'_NORM='+str(float(np.linalg.norm(x))))
 ra,rb=qa-qc,qb-qc;m('Q_RESPONSE_A='+v(ra));m('Q_RESPONSE_B='+v(rb));m('Q_RESPONSE_A_NORM='+str(float(np.linalg.norm(ra))));m('Q_RESPONSE_B_NORM='+str(float(np.linalg.norm(rb))));m('Q_RESPONSE_COSINE_A_B='+str(float(np.dot(ra,rb)/(np.linalg.norm(ra)*np.linalg.norm(rb)))))
 for n,x in [('A',ea),('B',eb),('C',ec)]:m('EEF_DELTA_'+n+'='+v(x))
 ra,rb=ea-ec,eb-ec;m('EEF_RESPONSE_A='+v(ra));m('EEF_RESPONSE_B='+v(rb));m('EEF_RESPONSE_A_NORM='+str(float(np.linalg.norm(ra))));m('EEF_RESPONSE_B_NORM='+str(float(np.linalg.norm(rb))));m('EEF_RESPONSE_COSINE_A_B='+str(float(np.dot(ra,rb)/(np.linalg.norm(ra)*np.linalg.norm(rb)))))
 m('CONTROL_OUTPUT_SEMANTICS=absolute_joint_position_target_after_compute_control_and_motor_limit_clip');m('MIN_JOINT_LIMIT_MARGIN=NOT_MEASURED');m('LEFT_Z_ROOT_CAUSE=UNRESOLVED');m('NEXT_STEP=TRACE_REVIEW');m('SHUTDOWN_BEGIN');og.shutdown()
except BaseException:traceback.print_exc();m('PYTHON_EXCEPTION=YES');raise
