from __future__ import annotations
import numpy as np

def sample_noise(shape, *, correlated=False, correlation=0.9, seed=None, dtype=np.float32, mask=None):
    if len(shape) != 3 or not all(int(x)>0 for x in shape): raise ValueError('shape must be [B,T,D]')
    if not 0 <= correlation < 1: raise ValueError('correlation must be in [0,1)')
    rng=np.random.default_rng(seed); out=np.empty(shape,dtype=dtype); out[:,0]=rng.standard_normal((shape[0],shape[2]))
    for t in range(1,shape[1]): out[:,t] = correlation*out[:,t-1] + np.sqrt(1-correlation**2)*rng.standard_normal((shape[0],shape[2])) if correlated else rng.standard_normal((shape[0],shape[2]))
    if mask is not None: out=np.where(np.asarray(mask,dtype=bool),out,0)
    return out
