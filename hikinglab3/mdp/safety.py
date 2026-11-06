# SPDX-License-Identifier: CC-BY-NC-4.0
"""Foot volume points and original edge-cylinder penetration cost."""

from types import SimpleNamespace
import torch
from isaaclab.utils.math import quat_apply
from hikinglab3.edges import GreedyconcatEdgeCylinder


def volume_points_penetration(env):
    if not hasattr(env, "_edge_detector"):
        cfg = SimpleNamespace(
            angle_threshold=70.0,
            adjacent_angle_threshold=30.0,
            min_points=2,
            cylinder_radius=0.05,
            num_grid_cells=64**3,
        )
        env._edge_detector = GreedyconcatEdgeCylinder(cfg)
        env._edge_detector.generate(env.scene.terrain.edge_mesh, device=env.device)
        env._foot_ids = env.scene["robot"].find_bodies(".*_ankle_roll_link")[0]
        env._foot_points = torch.stack(
            torch.meshgrid(
                torch.linspace(-0.025, 0.12, 10, device=env.device),
                torch.linspace(-0.03, 0.03, 5, device=env.device),
                torch.linspace(-0.063, -0.023, 2, device=env.device),
                indexing="ij",
            ),
            -1,
        ).reshape(-1, 3)
    robot = env.scene["robot"]
    ids = env._foot_ids
    q = robot.data.body_quat_w.torch[:, ids, None, :].expand(-1, -1, 100, -1)
    rel = quat_apply(q, env._foot_points.expand(env.num_envs, 2, -1, -1))
    points = robot.data.body_pos_w.torch[:, ids, None, :] + rel
    vel = robot.data.body_lin_vel_w.torch[:, ids, None, :] + torch.cross(
        robot.data.body_ang_vel_w.torch[:, ids, None, :].expand_as(rel), rel, dim=-1
    )
    penetration = env._edge_detector.get_points_penetration_offset(points.reshape(-1, 3)).reshape_as(points)
    return (penetration.norm(dim=-1) * (vel.norm(dim=-1) + 1e-6)).sum((1, 2))
