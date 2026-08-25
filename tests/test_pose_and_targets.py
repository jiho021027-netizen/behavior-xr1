import unittest

import numpy as np

from behavior_xr1.adapters import BehaviorStatePacker, r1pro_gripper_width_sum
from behavior_xr1.data.target_builder import (
    behavior_action_mask,
    build_xr1_action_horizon,
    build_xr1_action_target,
)
from behavior_xr1.geometry import (
    Pose,
    quaternion_xyzw_to_matrix,
    relative_pose_delta,
    relative_rotation_axis_angle,
    relative_translation_local,
)


def rotation_z(angle: float) -> np.ndarray:
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )


def state() -> np.ndarray:
    row = np.zeros(61, dtype=np.float64)
    row[20:24] = [0.0, 0.0, 0.0, 1.0]
    row[45:49] = [0.0, 0.0, 0.0, 1.0]
    return row


class PoseAndTargetTests(unittest.TestCase):
    def test_identity_pose_delta_is_zero(self):
        pose = Pose(np.zeros(3), np.eye(3))
        delta = relative_pose_delta(pose, pose)
        np.testing.assert_allclose(delta.translation, 0.0)
        np.testing.assert_allclose(delta.axis_angle, 0.0)

    def test_identity_orientation_world_translation_is_local_translation(self):
        np.testing.assert_allclose(
            relative_translation_local(np.zeros(3), np.eye(3), np.array([1.0, -2.0, 3.0])),
            [1.0, -2.0, 3.0],
        )

    def test_world_translation_is_rotated_into_eef_local_frame(self):
        np.testing.assert_allclose(
            relative_translation_local(np.zeros(3), rotation_z(np.pi / 2), np.array([0.0, 1.0, 0.0])),
            [1.0, 0.0, 0.0],
            atol=1e-7,
        )

    def test_pure_rotation_returns_axis_angle(self):
        np.testing.assert_allclose(
            relative_rotation_axis_angle(np.eye(3), rotation_z(np.pi / 2)),
            [0.0, 0.0, np.pi / 2],
            atol=1e-7,
        )

    def test_combined_pose_transform_and_xyzw_quaternion(self):
        current = Pose(np.array([1.0, 2.0, 0.0]), rotation_z(np.pi / 2))
        future = Pose(np.array([1.0, 3.0, 0.0]), rotation_z(np.pi))
        delta = relative_pose_delta(current, future)
        np.testing.assert_allclose(delta.translation, [1.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(delta.axis_angle, [0.0, 0.0, np.pi / 2], atol=1e-7)
        np.testing.assert_allclose(quaternion_xyzw_to_matrix([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]), rotation_z(np.pi / 2), atol=1e-7)

    def test_target_and_horizon_reconstruct_arm_and_trunk_deltas(self):
        current = state()
        future = state()
        future[17:20] = [0.2, 0.0, 0.0]
        future[42:45] = [0.0, 0.3, 0.0]
        future[45:49] = [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
        future[0:3] = [0.4, -0.2, 0.1]
        current[53:57] = [1.0, 2.0, 3.0, 4.0]
        future[53:57] = [1.5, 1.0, 4.0, 3.0]
        packed = build_xr1_action_target(current, future)
        self.assertEqual(packed.values.shape, (60,))
        np.testing.assert_allclose(packed.values[0:3], [0.2, 0.0, 0.0])
        np.testing.assert_allclose(packed.values[8:11], [0.0, 0.3, 0.0])
        np.testing.assert_allclose(packed.values[11:14], [0.0, 0.0, np.pi / 2], atol=1e-6)
        np.testing.assert_allclose(packed.values[17:20], future[0:3])
        np.testing.assert_allclose(packed.values[20:24], [0.5, -1.0, 1.0, -1.0])
        horizon = build_xr1_action_horizon(current, np.stack((future, future)))
        self.assertEqual(horizon.values.shape, (2, 60))
        self.assertEqual(horizon.mask.shape, (2, 60))

    def test_mask_excludes_gripper_waist_and_reserved_dimensions(self):
        mask = behavior_action_mask()
        self.assertEqual(mask.shape, (60,))
        self.assertTrue(mask[0:6].all())
        self.assertTrue(mask[8:14].all())
        self.assertTrue(mask[17:24].all())
        self.assertFalse(mask[6])
        self.assertFalse(mask[14])
        self.assertFalse(mask[16])
        self.assertFalse(mask[24:60].any())

    def test_invalid_state_and_unknown_gripper_are_explicit(self):
        with self.assertRaises(ValueError):
            build_xr1_action_target(np.zeros(60), np.zeros(61))
        with self.assertRaises(NotImplementedError):
            BehaviorStatePacker().pack(np.zeros(61))
        with self.assertRaises(ValueError):
            BehaviorStatePacker(lambda pair: pair).pack(np.zeros(61))

    def test_state_packer_requires_callback_and_packs_known_slices(self):
        source = state()
        source[3:10] = np.arange(7)
        source[28:35] = np.arange(10, 17)
        source[24:26] = [2.0, 4.0]
        source[49:51] = [3.0, 5.0]
        source[53:57] = [6.0, 7.0, 8.0, 9.0]
        source[0:3] = [0.1, 0.2, 0.3]
        packed = BehaviorStatePacker(lambda pair: pair[0]).pack(source)
        self.assertEqual(packed.shape, (60,))
        np.testing.assert_allclose(packed[0:7], source[3:10])
        np.testing.assert_allclose(packed[7], 2.0)
        np.testing.assert_allclose(packed[8:15], source[28:35])
        np.testing.assert_allclose(packed[15], 3.0)
        np.testing.assert_allclose(packed[16:20], source[53:57])
        np.testing.assert_allclose(packed[20:23], source[0:3])

    def test_source_backed_gripper_width_sum_is_explicit(self):
        self.assertEqual(r1pro_gripper_width_sum(np.array([0.02, 0.03])), 0.05)
        with self.assertRaises(ValueError):
            r1pro_gripper_width_sum(np.zeros(1))


if __name__ == "__main__":
    unittest.main()
