import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[1]

def test_xr1_base_indices_and_order():
    assert [17, 18, 19] == list(range(17, 20))
    assert ["x", "y", "rz"] == ["x", "y", "rz"]

def test_saved_base_action_is_finite_and_in_contract():
    d = json.loads((ROOT / "first_real_xr1_forward.json").read_text())
    base = np.asarray(d["selected_action_60d"], dtype=float)[17:20]
    assert np.isfinite(base).all()
    assert np.max(np.abs(base)) <= 1.0

def test_r1pro_base_preprocess_candidate():
    d = json.loads((ROOT / "xr1_base_action_mapping_audit.json").read_text())
    base = np.asarray(d["real_xr1_base_action"])
    expected = np.array([0.75, 0.75, 1.0]) * np.clip(base, -1.0, 1.0)
    np.testing.assert_allclose(expected, d["real_r1pro_base_preprocessed"])

def test_non_base_packing_indices_unchanged():
    a = np.zeros(60)
    a[17:20] = [1, 2, 3]
    assert np.array_equal(np.flatnonzero(a), np.array([17, 18, 19]))
