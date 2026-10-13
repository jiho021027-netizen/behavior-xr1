import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
EP = ROOT / "data/behavior_demo_validation/task-0000/episode_00000010"


def test_real_demo_manifest_is_official_r1pro_and_integrity_verified():
    result = json.loads((ROOT / "xr1_r1pro_real_demo_validation.json").read_text())
    assert result["robot_type"] == "R1Pro"
    assert result["timestamp_monotonic"] is True
    assert result["nan_inf_count"] == 0
    assert result["real_arm_target_roundtrip_valid"] is True
    assert result["camera_contract_valid"] is True


def test_real_trainer_sample_shapes_and_masks():
    sample = json.loads((EP / "r1pro_real_sample_validation.json").read_text())
    with np.load(EP / "r1pro_real_sample_validation.npz") as data:
        assert data["state"].shape == (60,)
        assert data["action_chunk"].shape == (30, 60)
        assert data["action_mask"].shape == (30, 60)
        assert np.all(data["action_mask"][:, 0:7] == 1)
        assert np.all(data["action_mask"][:, 8:15] == 1)
        assert np.all(data["action_mask"][:, 16:] == 0)
    assert sample["state_shape"] == [60]
    assert sample["action_shape"] == [30, 60]


def test_real_demo_camera_streams_are_three_rgb_slots():
    result = json.loads((ROOT / "xr1_r1pro_real_demo_validation.json").read_text())
    info = result["video_info"]
    assert set(info) == {"head", "left_wrist", "right_wrist"}
    assert info["head"]["shape_rgb"] == [720, 720, 3]
    assert info["left_wrist"]["shape_rgb"] == [480, 480, 3]
    assert info["right_wrist"]["shape_rgb"] == [480, 480, 3]
