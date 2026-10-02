from __future__ import annotations
import numpy as np
from .frame_transform import eef_local_to_base
class XR1EEFActionAdapter:
 """XR-1 EEF action adapter. Rotations are active world-frame matrices:
 R_world_from_eef and R_world_from_base; local translation uses
 R_base_from_eef = R_world_from_base.T @ R_world_from_eef.
 """
 def __init__(self, slices, *, fixed_trunk=True, neutral_unvalidated=True):
  self.slices=tuple(slices); self.fixed_trunk=fixed_trunk; self.neutral_unvalidated=neutral_unvalidated
  if not fixed_trunk: raise ValueError('semantic XR-1 waist→R1Pro trunk mapping is unavailable')
  if sum(s.command_dim for s in self.slices)!=21: raise ValueError('expected runtime 21D controller schema')
 def convert(self, action, rotation_left, rotation_right, *, base=None, trunk=None, gripper=None):
  a=np.asarray(action,dtype=np.float32)
  if a.shape[-1]!=60: raise ValueError('XR-1 action must end in 60')
  if self.neutral_unvalidated and not np.array_equal(a[...,17:20],np.zeros_like(a[...,17:20])): raise ValueError('UNVALIDATED_BASE_NONZERO')
  if self.neutral_unvalidated and (not np.array_equal(a[...,6:7],np.zeros_like(a[...,6:7])) or not np.array_equal(a[...,14:15],np.zeros_like(a[...,14:15]))): raise ValueError('UNVALIDATED_GRIPPER_NONZERO')
  out=np.zeros(a.shape[:-1]+(21,),np.float32)
  by={s.name:s for s in self.slices}
  def put(name,v): out[...,by[name].start:by[name].stop]=v
  put('base',a[...,17:20] if base is None else base)
  if trunk is None: raise ValueError('fixed-trunk mode requires current 4D trunk target')
  put('trunk',trunk)
  R_world_from_eef=np.asarray(rotation_left,dtype=np.float32)
  R_world_from_base=np.asarray(rotation_right,dtype=np.float32)
  R_base_from_eef=np.matmul(np.swapaxes(R_world_from_base,-1,-2),R_world_from_eef)
  put('arm_left',np.concatenate([eef_local_to_base(a[...,0:3],R_base_from_eef), a[...,3:6]],axis=-1)); put('arm_right',np.concatenate([eef_local_to_base(a[...,8:11],rotation_right), a[...,11:14]],axis=-1))
  if gripper is None: raise ValueError('explicit gripper hold/current command required')
  put('gripper_left',gripper[...,0:1]); put('gripper_right',gripper[...,1:2])
  if not np.isfinite(out).all(): raise ValueError('non-finite action')
  return out
