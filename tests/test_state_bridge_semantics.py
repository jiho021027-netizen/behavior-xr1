import numpy as np
import pytest
from behavior_xr1.adapters.state import BehaviorStatePacker

def test_state_bridge_preserves_xr1_compose_state_reserved_zero_tail():
    state=np.arange(61,dtype=np.float32)
    out=BehaviorStatePacker(lambda q: q.sum()).pack(state)
    assert out.shape==(60,)
    assert np.all(out[16:]==0)
    assert np.isfinite(out).all()

def test_state_bridge_rejects_wrong_shape_and_nonfinite_gripper_reduction():
    with pytest.raises(ValueError): BehaviorStatePacker(lambda q: q.sum()).pack(np.zeros(60))
    with pytest.raises(ValueError): BehaviorStatePacker(lambda q: float('nan')).pack(np.zeros(61))
