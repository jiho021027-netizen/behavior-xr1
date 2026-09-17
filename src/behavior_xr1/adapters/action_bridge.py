from __future__ import annotations
import numpy as np
from .frame_transform import eef_local_to_base
class XR1EEFActionAdapter:
 """Strict XR-1 EEF action to validated 21D IK vector; trunk is explicit hold."""
 def __init__(self, slices, *, fixed_trunk=True):
  self.slices=tuple(slices); self.fixed_trunk=fixed_trunk
  if not fixed_trunk: raise ValueError('semantic XR-1 waist→R1Pro trunk mapping is unavailable')
  if sum(s.command_dim for s in self.slices)!=21: raise ValueError('expected runtime 21D controller schema')
 def convert(self, action, rotation_left, rotation_right, *, base=None, trunk=None, gripper=None):
  a=np.asarray(action,dtype=np.float32)
  if a.shape[-1]!=60: raise ValueError('XR-1 action must end in 60')
  out=np.zeros(a.shape[:-1]+(21,),np.float32)
  by={s.name:s for s in self.slices}
  def put(name,v): out[...,by[name].start:by[name].stop]=v
  put('base',a[...,17:20] if base is None else base)
  if trunk is None: raise ValueError('fixed-trunk mode requires current 4D trunk target')
  put('trunk',trunk)
  put('arm_left',eef_local_to_base(a[...,0:3],rotation_left)); put('arm_right',eef_local_to_base(a[...,8:11],rotation_right))
  if gripper is None: raise ValueError('explicit gripper hold/current command required')
  put('gripper_left',gripper[...,0:1]); put('gripper_right',gripper[...,1:2])
  if not np.isfinite(out).all(): raise ValueError('non-finite action')
  return out
