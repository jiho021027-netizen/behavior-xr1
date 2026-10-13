from __future__ import annotations
from dataclasses import asdict,dataclass
import json
@dataclass(frozen=True)
class CheckpointManifest:
 behavior_xr1_commit:str; xr1_commit:str; behavior_version:str; task_map_hash:str; state_schema:str; action_schema:str; controller_strategy:str; normalization_schema:str; task_conditioning:dict; skill:dict; noise:dict
 def to_dict(self): return asdict(self)
 def save(self,path):
  with open(path,'w') as f: json.dump(self.to_dict(),f,indent=2,sort_keys=True)
 @classmethod
 def load(cls,path):
  with open(path) as f: return cls(**json.load(f))
 def require_compatible(self, expected):
  for k,v in expected.items():
   if getattr(self,k,None)!=v: raise ValueError(f'checkpoint manifest mismatch: {k}')
