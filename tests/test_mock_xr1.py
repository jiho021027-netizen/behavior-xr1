import numpy as np
from behavior_xr1.evaluation.mock_xr1 import neutral_xr1_action,neutral_bridge

def test_neutral_action_is_safe_and_bridges():
 a=neutral_xr1_action(); assert a.shape==(60,); assert not a.any(); assert not a[17:20].any(); assert a[6]==a[14]==0
 obs={'robot::proprio':np.zeros(61,dtype=np.float32)}; state,action,out=neutral_bridge(obs); assert state.shape==(60,); assert not state[16:].any(); assert out.shape==(21,); assert np.isfinite(out).all()
