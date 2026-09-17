from __future__ import annotations
import numpy as np
from .r1pro_action_schema import ControllerSlice

def build_hold_action(slices, *, trunk_target, gripper_target=(0.0,0.0)):
 """Build runtime-only fixed-trunk hold action for the validated 21D schema."""
 out=np.zeros(sum(s.command_dim for s in slices),dtype=np.float32); by={s.name:s for s in slices}
 if np.asarray(trunk_target).shape!=(4,): raise ValueError('trunk_target must be 4D current joint target')
 if np.asarray(gripper_target).shape!=(2,): raise ValueError('gripper_target must provide current left/right commands')
 for name,val in [('trunk',trunk_target),('gripper_left',[gripper_target[0]]),('gripper_right',[gripper_target[1]])]:
  s=by[name]; arr=np.asarray(val,dtype=np.float32).reshape(-1)
  if arr.size!=s.command_dim: raise ValueError(f'{name} target dimension mismatch')
  out[s.start:s.stop]=arr
 return out
