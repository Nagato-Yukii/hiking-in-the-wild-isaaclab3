# SPDX-License-Identifier: CC-BY-NC-4.0
"""Original temporal observations; 3.0 tensor views and camera pipeline."""

import torch
import torch.nn.functional as F
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.managers import ObservationGroupCfg as Group, ObservationTermCfg as Term
from isaaclab.envs import mdp


def depth_history(env):
    sensor = env.scene["camera"]
    raw = sensor.data.output["distance_to_image_plane"]
    raw = raw.torch if hasattr(raw, "torch") else raw
    raw = torch.nan_to_num(raw.squeeze(-1), nan=2.5, posinf=2.5).clamp(0.0, 2.5)
    if not hasattr(env, "_depth_ring"):
        env._depth_ring = torch.zeros(env.num_envs, 37, 18, 32, device=env.device)
        env._depth_step = -1
        env._depth_delay = torch.rand(env.num_envs, 1, device=env.device)
    if env._depth_step != env.common_step_counter:
        # Upstream CropAndResizeCfg uses (top,bottom,left,right), then resizes to (18,32).
        crop = raw[:, 18:, 16:-16].unsqueeze(1)
        crop = F.interpolate(crop, size=(18, 32), mode="bilinear", align_corners=False)
        k = torch.tensor([0.27406862, 0.45186276, 0.27406862], device=env.device)
        kernel = (k[:, None] * k[None, :]).reshape(1, 1, 3, 3)
        blurred = F.conv2d(F.pad(crop, (1, 1, 1, 1), mode="reflect"), kernel).squeeze(1) / 2.5
        env._depth_ring = torch.roll(env._depth_ring, -1, 1)
        env._depth_ring[:, -1] = blurred
        reset = env.episode_length_buf == 0
        env._depth_ring[reset] = blurred[reset, None]
        env._depth_step = env.common_step_counter
        env._depth_delay[reset] = torch.rand(int(reset.sum()), 1, device=env.device)
        indices = (36 - torch.arange(7, -1, -1, device=env.device)[None] * 5 - env._depth_delay).long()
        env._depth_output = env._depth_ring[torch.arange(env.num_envs, device=env.device)[:, None], indices]
    return env._depth_output


@configclass
class Policy(Group):
    base_ang_vel = Term(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=8)
    projected_gravity = Term(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), history_length=8)
    velocity_commands = Term(func=mdp.generated_commands, params={"command_name": "base_velocity"}, history_length=8)
    joint_pos = Term(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=8)
    joint_vel = Term(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=8)
    actions = Term(func=mdp.last_action, history_length=8)
    depth_image = Term(func=depth_history)

    def __post_init__(self):
        self.concatenate_terms = False
        self.enable_corruption = True


@configclass
class Critic(Policy):
    base_lin_vel = Term(func=mdp.base_lin_vel, history_length=8)

    def __post_init__(self):
        self.concatenate_terms = False
        self.enable_corruption = False


@configclass
class AmpPolicy(Group):
    projected_gravity = Term(func=mdp.projected_gravity, history_length=10)
    joint_pos_rel = Term(func=mdp.joint_pos_rel, history_length=10)
    joint_vel = Term(func=mdp.joint_vel_rel, scale=0.05, history_length=10)
    base_lin_vel = Term(func=mdp.base_lin_vel, history_length=10)
    base_ang_vel = Term(func=mdp.base_ang_vel, history_length=10)

    def __post_init__(self):
        self.concatenate_terms = False
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    policy = Policy()
    critic = Critic()
    amp_policy = AmpPolicy()
