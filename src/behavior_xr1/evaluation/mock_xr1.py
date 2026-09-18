from __future__ import annotations
import numpy as np
from behavior_xr1.adapters import BehaviorStatePacker, r1pro_gripper_width_sum, XR1EEFActionAdapter, build_controller_slices

def neutral_xr1_action(): return np.zeros(60,dtype=np.float32)
def bridge_observation(obs):
 def flat(x,p=''):
  if isinstance(x,dict):
   for k,v in x.items(): yield from flat(v,f'{p}.{k}' if p else k)
  else: yield p,x
 candidates=[np.asarray(v) for k,v in flat(obs) if np.asarray(v).shape[-1:]==(61,)]
 if len(candidates)!=1: raise ValueError(f'expected one 61D proprioception item, got {len(candidates)}')
 state=BehaviorStatePacker(r1pro_gripper_width_sum).pack(candidates[0])
 if not np.isfinite(state).all() or not np.all(state[...,16:]==0): raise ValueError('invalid XR1 state bridge')
 return state,candidates[0]
def neutral_bridge(obs):
 state,prop=bridge_observation(obs); action=neutral_xr1_action()
 slices=build_controller_slices(('base','trunk','arm_left','gripper_left','arm_right','gripper_right'),(3,4,6,1,6,1),21)
 out=XR1EEFActionAdapter(slices).convert(action,np.eye(3),np.eye(3),trunk=np.asarray(prop)[...,53:57],gripper=np.zeros(np.asarray(prop).shape[:-1]+(2,),np.float32))
 return state,action,out
