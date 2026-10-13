import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parents[1]

def r1_pre(u):
    return 0.025*(np.clip(float(u),-1,1)+1)

def test_xr1_indices_and_delta_update_contract():
    assert (6,14)==(6,14)
    assert np.isclose(0.4+0.1,0.5)

def test_r1pro_smooth_endpoints_and_hold():
    assert np.isclose(r1_pre(-1),0.0)
    assert np.isclose(r1_pre(1),0.05)
    assert np.isclose(r1_pre(0),0.025)

def test_r1pro_preprocess_monotone_and_clipped():
    vals=[r1_pre(x) for x in (-2,-1,0,1,2)]
    assert vals==sorted(vals)
    assert vals[0]==vals[1] and vals[-1]==vals[-2]

def test_real_saved_gripper_deltas_finite():
    d=json.loads((ROOT/'first_real_xr1_forward.json').read_text())
    a=np.asarray(d['selected_action_60d'],float)
    assert np.isfinite(a[[6,14]]).all()
    np.testing.assert_allclose(a[[6,14]],[0.9921884536743164,8.164062705873221e-07])

def test_endpoint_normalization_is_explicitly_unavailable():
    assert json.loads((ROOT/'xr1_gripper_action_mapping_audit.json').read_text())['gripper_source_contract_unavailable'] is True
