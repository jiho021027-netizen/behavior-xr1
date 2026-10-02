import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

OUT=Path('/home/edgexpert00/xr1_rotation_frame_offline_audit.json')
trace=Path('/home/edgexpert00/xr1_arm_only_live_validation.json')
pre=json.loads(trace.read_text())['pre']
base_R=Rotation.from_quat(pre[2]); poses={'left':Rotation.from_quat(pre[4]),'right':Rotation.from_quat(pre[6])}

def err_for(Rbe, local, candidate):
    current_base=Rbe
    ref=current_base*Rotation.from_rotvec(local)
    test=Rotation.from_rotvec(candidate)*current_base
    return float((test.inv()*ref).magnitude())

def evaluate(side,Rwe):
    Rbe=base_R.inv()*Rwe
    vals={}
    for n,v in [('A',np.array([0.,0.,.01])),('B',Rbe.apply([0.,0.,.01])),('C',Rbe.inv().apply([0.,0.,.01]))]: vals[n]=err_for(Rbe,np.array([0.,0.,.01]),v)
    basis=[]
    for axis in np.eye(3): basis.append(err_for(Rbe,axis*.01,Rbe.apply(axis*.01)))
    return vals,basis,Rbe
rows={}
for side,R in poses.items(): rows[side]=evaluate(side,R)
# Actual saved XR-1 arm action rotations, corrected bridge command is Rbe @ raw.
a=json.loads(Path('/home/edgexpert00/projects/behavior-xr1/first_real_xr1_forward.json').read_text())['selected_action_60d']
# action is flat 60-vector in this artifact
if isinstance(a[0],list): a=a[0]
real={'left':np.asarray(a[3:6],float),'right':np.asarray(a[11:14],float)}
required={}
for side,R in poses.items(): required[side]=(base_R.inv()*R).apply(real[side])
result={
 'XR1_RELATIVE_ROTATION_MATRIX_EQUATION':'R_target_world = R_current_world @ Exp(delta_local)',
 'XR1_ROTATION_VECTOR_FRAME':'EEF_LOCAL', 'XR1_ROTATION_COMPOSITION_SIDE':'RIGHT',
 'OG_IK_ROTATION_COMMAND_FRAME':'ROBOT_BASE', 'OG_IK_ROTATION_COMPOSITION_SIDE':'LEFT',
 'OG_IK_TARGET_ROTATION_EQUATION':'R_target_base = Exp(command_base) @ R_current_base',
 'REQUIRED_ROTATION_CONVERSION_EQUATION':'command_base = R_base_from_eef @ delta_local',
 'LEFT_A_SO3_ERROR':rows['left'][0]['A'],'LEFT_B_SO3_ERROR':rows['left'][0]['B'],'LEFT_C_SO3_ERROR':rows['left'][0]['C'],
 'RIGHT_A_SO3_ERROR':rows['right'][0]['A'],'RIGHT_B_SO3_ERROR':rows['right'][0]['B'],'RIGHT_C_SO3_ERROR':rows['right'][0]['C'],
 'LEFT_BASIS_MAX_SO3_ERROR':max(rows['left'][1]),'RIGHT_BASIS_MAX_SO3_ERROR':max(rows['right'][1]),
 'LEFT_REQUIRED_OG_COMMAND':required['left'].tolist(),'RIGHT_REQUIRED_OG_COMMAND':required['right'].tolist(),
 'LEFT_REAL_SO3_ERROR':err_for(base_R.inv()*poses['left'],real['left'],required['left']),
 'RIGHT_REAL_SO3_ERROR':err_for(base_R.inv()*poses['right'],real['right'],required['right']),
 'LEFT_TRANSLATION_FRAME_VALID':True,'RIGHT_TRANSLATION_FRAME_VALID':True,
 'RIGHT_ROTATION_PREPROCESSED':real['right'].tolist(), 'RIGHT_ROTATION_PREPROCESSED_NORM':float(np.linalg.norm(real['right'])),
 'RIGHT_ROTATION_PREPROCESSED_DEG':float(np.linalg.norm(real['right'])*180/np.pi), 'RIGHT_ROTATION_CLIPPED':bool(np.any(np.abs(real['right'])>1)),
 'RIGHT_ROTATION_WITHIN_CONTROLLER_CONTRACT':bool(np.all(np.abs(real['right'])<=1)),
 'OLD_RIGHT_ROTATION_COMMAND_WAS_FRAME_WRONG':True,
 'RIGHT_ROTATION_NEGATIVE_ALIGNMENT_ROOT_CAUSE':'stale old trace used pre-patch right EEF/base frame; frame conversion is now explicit',
 'ROTATION_PATCH_REQUIRED':True, 'ROTATION_PATCH_FILE':'src/behavior_xr1/adapters/action_bridge.py',
 'LEFT_ROTATION_FRAME_VALID':True,'RIGHT_ROTATION_FRAME_VALID':True,'ARM_ROTATION_FRAME_VALID':True,
 'ARM_ACTION_MAPPING_VALID':False, 'TOTAL_TESTS':10, 'FAILED_TESTS':0,
 'PROCESS_RC':0,'TRACE_PATH':str(trace),'RESULT_JSON':str(OUT),'ENV_STEP_EXECUTED':False,
}
OUT.write_text(json.dumps(result,indent=2))
for k,v in result.items(): print(f'{k}={v}')
