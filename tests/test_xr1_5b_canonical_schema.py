import json
from pathlib import Path


def test_generic_5b_schema_has_nonoverlapping_bimanual_fields():
    p = Path(__file__).parents[1] / "xr1_5b_canonical_action_schema.json"
    schema = json.loads(p.read_text())["schema"]
    semantic = [f for f in schema["fields"] if f["name"] not in {"reserved"}]
    used = []
    for field in semantic:
        start, end = (int(x) for x in field["indices"].split(":"))
        assert not set(range(start, end)) & set(used)
        used.extend(range(start, end))
    assert {f["name"] for f in semantic} >= {
        "left_ee_pos", "left_ee_aa", "left_gripper",
        "right_ee_pos", "right_ee_aa", "right_gripper",
    }
    assert len(used) == 18


def test_5b_generic_mask_and_state_mask_are_distinct_and_explicit():
    audit = json.loads((Path(__file__).parents[1] / "xr1_5b_bimanual_contract_audit.json").read_text())
    assert audit["bimanual_contract_valid"] is True
    assert audit["r1pro_direct_contract_valid"] is False
    assert audit["action_norm_active_indices"] == [0,1,2,3,4,5,6,8,9,10,11,12,13,14,16,17,18,19]
    assert audit["state_norm_active_indices"] == [0,1,2,3,4,5,7,8,9,10,11,12,13,15]
