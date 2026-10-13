from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class ControllerSlice:
 name:str; start:int; stop:int; command_dim:int

def build_controller_slices(order, command_dims, expected_dim=None):
 if len(order)!=len(command_dims): raise ValueError('order/dim length mismatch')
 out=[]; off=0
 for n,d in zip(order,command_dims):
  if int(d)<=0: raise ValueError(f'invalid command_dim for {n}')
  out.append(ControllerSlice(str(n),off,off+int(d),int(d))); off+=int(d)
 if expected_dim is not None and off!=expected_dim: raise ValueError(f'controller dims sum {off} != expected {expected_dim}')
 return tuple(out)
