import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1]
UP=np.array([0.45,-0.4,0.0,0.0]); DOWN=np.array([1.6,-2.5,-0.94,0.0]); GROUND=np.array([1.735,-2.57,-2.1,0.0])
def manifold(s):
    s=float(np.clip(s,0,2))
    return (1-s)*UP+s*DOWN if s<=1 else (2-s)*DOWN+(s-1)*GROUND
def test_waist_action_index_and_source_block():
    d=json.loads((ROOT/'xr1_trunk_action_mapping_audit.json').read_text())
    assert d['xr1_waist_action_index']==16 and d['trunk_source_contract_unavailable'] is True
def test_existing_manifold_endpoints():
    np.testing.assert_allclose(manifold(0),UP); np.testing.assert_allclose(manifold(1),DOWN); np.testing.assert_allclose(manifold(2),GROUND)
def test_manifold_monotonic_piecewise_and_finite():
    vals=np.array([manifold(x) for x in np.linspace(0,2,21)])
    assert np.isfinite(vals).all(); assert np.all(np.diff(vals[:,0])>=-1e-9); assert np.all(np.diff(vals[:,1])<=1e-9)
def test_repeated_delta_reversibility_in_scalar_coordinate():
    s=0.7; s2=np.clip(s+0.1,0,2); s3=np.clip(s2-0.1,0,2); np.testing.assert_allclose(manifold(s3),manifold(s))
def test_real_waist_action_is_exact_saved_value():
    d=json.loads((ROOT/'first_real_xr1_forward.json').read_text()); assert np.isclose(d['selected_action_60d'][16],-3.2617188594485924e-07)
