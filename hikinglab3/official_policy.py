"""Exact feedforward operations from the released stair ONNX files, batched on Torch."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# PhysX articulation order observed for the identical torso-base G1 shoe URDF.
PHYSX_JOINTS = [
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "waist_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "waist_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "waist_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint", "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
]

class OfficialStairPolicy:
    def __init__(self, joint_names, device):
        root = Path(__file__).resolve().parents[1] / "artifacts/official_stair_adapter"
        with np.load(root / "weights.npz") as f:
            self.w = {k: torch.tensor(f[k], device=device) for k in f.files}
        assert set(joint_names) == set(PHYSX_JOINTS)
        self.to_official = torch.tensor([joint_names.index(n) for n in PHYSX_JOINTS], device=device)
        self.to_native = torch.tensor([PHYSX_JOINTS.index(n) for n in joint_names], device=device)
        with np.load(root / "reference_vectors.npz") as f, torch.inference_mode():
            z = self.encode(torch.tensor(f['depth'], device=device))
            a = self.actor(torch.cat((torch.tensor(f['proprio'], device=device), z), dim=1))
            self.validation = {
                'encoder_max_abs_error': float((z - torch.tensor(f['encoded'], device=device)).abs().max()),
                'actor_max_abs_error': float((a - torch.tensor(f['actions'], device=device)).abs().max()),
                'reference': 'onnx.reference.ReferenceEvaluator, 5 batch-one inputs; Torch batched execution',
            }
            assert self.validation['encoder_max_abs_error'] < 1e-3
            assert self.validation['actor_max_abs_error'] < 1e-3

    def linear(self, x, prefix):
        return F.linear(x, self.w[prefix + '.weight'], self.w[prefix + '.bias'])

    def encode(self, depth):
        x = F.relu(F.conv2d(depth, self.w['encoder::conv.conv.0.weight'],
                           self.w['encoder::conv.conv.0.bias'], padding=1)).flatten(1)
        for i in (0, 2, 4):
            x = F.relu(self.linear(x, f'encoder::head.model.{i}'))
        return x

    def actor(self, x):
        for i in (0, 2, 4):
            x = F.elu(self.linear(x, f'actor::{i}'))
        return self.linear(x, 'actor::6')

    def __call__(self, obs):
        # Each term is separately history-major, not a history of concatenated terms.
        proprio = [obs[:, :72]]
        for start in (72, 304, 536):
            proprio.append(obs[:, start:start+232].reshape(-1, 8, 29)[:, :, self.to_official].flatten(1))
        z = self.encode(obs[:, 768:].reshape(-1, 8, 18, 32))
        actions = self.actor(torch.cat((*proprio, z), dim=1))
        return actions[:, self.to_native]
