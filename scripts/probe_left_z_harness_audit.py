import copy
import traceback
import numpy as np
from omegaconf import OmegaConf
import omnigibson as og
from omnigibson.eval.evaluator import Evaluator
from omnigibson.controllers import ControllerView
from behavior_xr1.adapters import XR1EEFActionAdapter, build_controller_slices
from behavior_xr1.geometry.pose import quaternion_xyzw_to_matrix

def m(s): print(s, flush=True)
def as_np(x):
    if hasattr(x, 'detach'): x = x.detach().clone().cpu().numpy()
    elif hasattr(x, 'numpy'): x = x.numpy()
    return np.array(x, dtype=float, copy=True)
def snap(x): return as_np(x)
def fmt(x): return str(snap(x).tolist())
def ptr(x): return str(x.data_ptr()) if hasattr(x, 'data_ptr') else 'NOT_TENSOR'
def pose(x):
    p, q = x.get_position_orientation()
    return as_np(p), as_np(q)
def flat(x):
    if isinstance(x, dict):
        for y in x.values(): yield from flat(y)
    else: yield x
def cfg():
    r=OmegaConf.load('/home/edgexpert00/projects/behavior-xr1/configs/eval/r1pro_xr1_ik.yaml')
    return OmegaConf.create({'env_wrapper':{'_target_':'omnigibson.eval.wrappers.DefaultWrapper'},'policy_name':'local','model':{'_target_':'omnigibson.eval.policies.LocalPolicy','action_dim':None},'headless':True,'partial_scene_load':True,'max_steps':3,'write_video':False,'mode':'public_test','seed':0,'task':{'name':'turning_on_radio'},'robot':r})
D={'X':(.01,0,0),'Y':(0,.01,0),'Z':(0,0,.01),'C':(0,0,0)}
try:
    e=Evaluator(cfg()); e.reset(); e.load_task_instance(301)
    def run(seq_name, sequence):
        e.reset()
        b=e.robot; group, idx=b.controllers['arm_left']; ctl=ControllerView._controller_groups[group]
        ep, eq=pose(b.eef_links['left']); _, bq=pose(b)
        R=quaternion_xyzw_to_matrix(bq).T @ quaternion_xyzw_to_matrix(eq)
        prop=next(np.asarray(z) for z in flat(e.obs) if np.asarray(z).shape[-1:]==(61,))
        bridge=XR1EEFActionAdapter(build_controller_slices(('base','trunk','arm_left','gripper_left','arm_right','gripper_right'),(3,4,6,1,6,1),21))
        saved={}
        for call_index, label in enumerate(sequence):
            raw60=np.zeros(60,np.float32); raw60[:3]=D[label]
            action=bridge.convert(raw60,R,R,trunk=prop[53:57],gripper=np.zeros(2,np.float32))
            command=action[7:13]
            m(f'CASE={seq_name}');m(f'CALL_INDEX={call_index}');m(f'EXPECTED_LABEL={label}');m(f'EXPECTED_RAW_COMMAND={list(D[label])}');m(f'ACTUAL_RAW_COMMAND={fmt(command)}');m('LABEL_COMMAND_MATCH='+str(bool(np.allclose(np.asarray(command)[:3],D[label]))))
            m('RAW_COMMAND_COPY='+fmt(command))
            # Exact production order, without ControllerView.step_all() / deployment.
            goal_before=copy.deepcopy(ctl.get_goal(idx))
            ctl.update_goal(idx, command)
            goal_after=copy.deepcopy(ctl.get_goal(idx))
            goals_used={k: v for k,v in ctl._goals.items()}
            pre=ctl._preprocess_command(command)
            m('PREPROCESSED_COMMAND_COPY='+fmt(pre))
            m('CONTROL_DICT_RELEVANT_FIELDS_COPY='+str({k:fmt(v) for k,v in goals_used.items()}))
            m('GOAL_BEFORE='+str({k:fmt(v) for k,v in goal_before.items()}))
            m('GOAL_AFTER_PREPARATION='+str({k:fmt(v) for k,v in goal_after.items()}))
            m('GOAL_USED_BY_COMPUTE_CONTROL='+str({k:fmt(v) for k,v in goals_used.items()}))
            m('GOAL_POS_COPY='+fmt(goals_used['target_pos'][idx]));m('GOAL_ORI_COPY='+fmt(goals_used['target_ori_mat'][idx]));m('GOAL_OBJECT_ID='+str(id(goals_used)))
            out=ctl.compute_control(goals_used)
            m('TARGET_OBJECT_ID='+str(id(out)));m('TARGET_DATA_PTR='+ptr(out));m('TARGET_IMMEDIATE_COPY='+fmt(out[idx]))
            saved[label]=snap(out[idx])
            same_goal=all(np.array_equal(snap(goals_used[k][idx]),snap(goal_after[k])) for k in goals_used)
            m('CURRENT_COMMAND_GOAL_USED='+str(same_goal))
        response=float(np.linalg.norm(saved['Z']-saved['C']))
        m(f'SEQ_{seq_name}_Z_RESPONSE_NORM={response}')
        return response
    results={'A':run('A','ZC'),'B':run('B','XZC'),'C':run('C','CZC')}
    m('SEQUENCE_DEPENDENCE='+str(results))
    m('LEFT_Z_ROOT_CAUSE=TRACE_REVIEW')
    m('NEXT_STEP=TRACE_REVIEW')
    m('SHUTDOWN_BEGIN'); og.shutdown()
except BaseException:
    traceback.print_exc();m('PYTHON_EXCEPTION=YES');raise
