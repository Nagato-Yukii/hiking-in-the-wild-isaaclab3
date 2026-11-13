"""Protect the dataset-to-policy joint order, units and reference-frame contract."""

import tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from hikinglab3.motions import MotionDataset


class MotionContract(unittest.TestCase):
    def test_unfiltered_reference_crosses_large_root_and_joint_transitions(self):
        with tempfile.TemporaryDirectory() as folder:
            frames = 40
            position = np.zeros((frames, 3), dtype=np.float32)
            position[20:, 0] = 12.0
            joints = np.zeros((frames, 2), dtype=np.float32)
            joints[20:, 0] = 3.0
            path = Path(folder) / 'motion.npz'
            np.savez(path, framerate=50., joint_names=['a','b'], joint_pos=joints,
                     base_pos_w=position,
                     base_quat_w=np.tile(np.array([1.,0.,0.,0.],dtype=np.float32),(frames,1)))
            full = MotionDataset(['a','b'],torch.zeros(1,2),'cpu',path)
            self.assertEqual(full.starts.tolist(),list(range(31)))
            full.advance(1,0,torch.ones(1,dtype=torch.bool))
            full._cursor.fill_(9)
            full._mirror_mask.fill_(False)
            for step in range(1,11):
                actual = full.advance(1,step,torch.zeros(1,dtype=torch.bool))
            self.assertEqual(full._cursor.item(),19)
            expected = torch.cat([v[10:20].flatten() for v in full.features.values()])[None]
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            self.assertEqual(full.features['base_lin_vel'][19,0].item(),600.)
            self.assertEqual(full.features['joint_vel'][19,0].item(),7.5)
            # Every source frame is an eligible reset cursor, including the tail.
            from unittest.mock import patch
            original_randint = torch.randint
            with patch('torch.randint',wraps=original_randint) as draw:
                full.advance(1,11,torch.ones(1,dtype=torch.bool))
                self.assertEqual(draw.call_args.args[0],frames)
            full._cursor.fill_(frames-2)
            full.advance(1,12,torch.zeros(1,dtype=torch.bool))
            self.assertEqual(full._cursor.item(),frames-1)
            full.advance(1,13,torch.zeros(1,dtype=torch.bool))
            self.assertLess(full._cursor.item(),frames)

    def test_wxyz_joint_mapping_and_body_velocity(self):
        with tempfile.TemporaryDirectory() as folder:
            t = np.arange(30, dtype=np.float32) / 50
            q = Rotation.from_euler("z", (np.pi / 2 + t)[:, None]).as_quat().astype(np.float32)
            path = Path(folder) / "motion.npz"
            np.savez(
                path,
                framerate=50.0,
                joint_names=["b", "a"],
                joint_pos=np.stack([2 * t, t], -1),
                base_pos_w=np.stack([t, t * 0, t * 0], -1),
                base_quat_w=q[:, [3, 0, 1, 2]],
            )
            data = MotionDataset(["a", "b"], torch.tensor([[0.2, 0.4]]), "cpu", path)
            torch.testing.assert_close(data.features["joint_pos_rel"][0], torch.tensor([-0.2, -0.4]))
            torch.testing.assert_close(data.features["joint_vel"][0], torch.tensor([0.05, 0.1]))
            torch.testing.assert_close(
                data.features["base_lin_vel"][0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                data.features["base_ang_vel"][0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5, rtol=1e-5
            )
            self.assertEqual(data.sample(5).shape, (5, 130))
            original = {key: value[:3].clone() for key, value in data.features.items()}
            mirror = data.mirror(original)
            torch.testing.assert_close(mirror["base_lin_vel"][:, 1], -original["base_lin_vel"][:, 1])
            torch.testing.assert_close(mirror["base_ang_vel"][:, 2], -original["base_ang_vel"][:, 2])
            for key, value in data.mirror(mirror).items():
                torch.testing.assert_close(value, original[key])
            reset = torch.ones(5, dtype=torch.bool)
            first = data.advance(5, 0, reset)
            first_copy = first.clone()
            first_cursor = data._cursor.clone()
            torch.testing.assert_close(data.advance(5, 0, reset), first_copy)
            torch.testing.assert_close(data._cursor, first_cursor)
            old_history = data._reference_history["joint_pos_rel"].clone()
            torch.testing.assert_close(old_history[:, 0], old_history[:, -1])
            # Keep this assertion away from reference-clip boundaries, which reset history.
            data._cursor.fill_(2)
            second = data.advance(5, 1, torch.zeros(5, dtype=torch.bool))
            torch.testing.assert_close(data._reference_history["joint_pos_rel"][:, :-1], old_history[:, 1:])
            self.assertEqual(first.shape, second.shape)
            torch.testing.assert_close(data.advance(5, 1, torch.zeros(5, dtype=torch.bool)), second)
            partial_reset = torch.tensor([True, False, False, False, False])
            data._cursor.fill_(2)
            mixed = data.advance(5, 2, partial_reset).clone()
            mixed_cursor = data._cursor.clone()
            torch.testing.assert_close(data.advance(5, 2, partial_reset), mixed)
            torch.testing.assert_close(data._cursor, mixed_cursor)
            # Explicit reset at the same clock step must still refresh all histories.
            data.reset()
            data.advance(5, 2, reset)
            for history in data._reference_history.values():
                torch.testing.assert_close(history[:, 0], history[:, -1])
            with self.assertRaisesRegex(ValueError, "Joint mismatch"):
                MotionDataset(["a", "missing"], torch.zeros(1, 2), "cpu", path)


if __name__ == "__main__":
    unittest.main()
