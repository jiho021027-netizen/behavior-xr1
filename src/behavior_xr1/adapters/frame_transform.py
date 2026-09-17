from __future__ import annotations
import numpy as np
def eef_local_to_base(delta_local, rotation_base_eef):
 d=np.asarray(delta_local,dtype=np.float32); R=np.asarray(rotation_base_eef,dtype=np.float32)
 if d.shape[-1]!=3 or R.shape[-2:]!=(3,3): raise ValueError('delta [...,3], rotation [...,3,3] required')
 out=np.einsum('...ij,...j->...i',R,d)
 if not np.isfinite(out).all(): raise ValueError('non-finite frame transform')
 return out
