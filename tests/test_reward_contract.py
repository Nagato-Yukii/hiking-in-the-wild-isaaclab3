"""Regression: a healthy motor moving below its rating must not be overspeed."""

import unittest
from types import SimpleNamespace as NS
import torch
from hikinglab3.mdp.regularizations import joint_vel_limits
from isaaclab.managers import SceneEntityCfg


class RewardContract(unittest.TestCase):
    def test_native_speed_ratings_not_legacy_zero_buffer(self):
        asset = NS(
            data=NS(
                joint_vel=NS(torch=torch.tensor([[5.0, -18.5, 22.0]])),
                joint_velocity_limits=NS(torch=torch.tensor([[20.0, 20.0, 20.0]])),
                soft_joint_vel_limits=NS(torch=torch.zeros(1, 3)),
            )
        )
        env = NS(scene={"robot": asset})
        # threshold 18 rad/s: costs 0 + .5 + clipped(4)=1.5.
        torch.testing.assert_close(joint_vel_limits(env, 0.9), torch.tensor([1.5]))
        cfg = SceneEntityCfg("robot", joint_ids=[0])
        torch.testing.assert_close(joint_vel_limits(env, 0.9, cfg), torch.tensor([0.0]))


class GravityContract(unittest.TestCase):
    def test_tilt_penalty_uses_direction_not_gravity_acceleration(self):
        from hikinglab3.mdp.rewards import link_orientation
        import math

        angle = math.pi / 6
        # A 30 degree roll: squared horizontal gravity direction is sin^2=0.25.
        quaternion = torch.tensor([[[math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)]]])
        asset = NS(data=NS(body_quat_w=NS(torch=quaternion), GRAVITY_VEC_W=NS(torch=torch.tensor([[0.0, 0.0, -9.81]]))))
        env = NS(scene={"robot": asset})
        cfg = SceneEntityCfg("robot", body_ids=[0])
        torch.testing.assert_close(link_orientation(env, cfg), torch.tensor([0.25]))
        asset.data.GRAVITY_VEC_W.torch *= 2
        torch.testing.assert_close(link_orientation(env, cfg), torch.tensor([0.25]))
