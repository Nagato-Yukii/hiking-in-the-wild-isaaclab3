# SPDX-License-Identifier: CC-BY-NC-4.0
"""Official retargeted NPZ to 10-frame AMP windows, with explicit joint/quaternion mapping."""

from pathlib import Path
import hashlib
import numpy as np
import torch
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_conjugate, axis_angle_from_quat

DEFAULT_MOTION = (
    Path(__file__).resolve().parents[1]
    / "data/official/data&model/parkour_motion_reference/parkour_motion_without_run_retargetted.npz"
)


class MotionDataset:
    def __init__(self, joint_names, default_joint_pos, device, path=DEFAULT_MOTION, *, filter_discontinuities=True):
        self.filter_discontinuities = filter_discontinuities
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        required = {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
        if not required.issubset(data.files):
            raise ValueError("Missing retargeted motion fields")
        self.fps = float(data["framerate"])
        self.frames = len(data["joint_pos"])
        if self.fps != 50.0:
            raise ValueError("This task requires 50 Hz motions; resample before use")
        names = data["joint_names"].tolist()
        if set(names) != set(joint_names):
            raise ValueError(f"Joint mismatch: {set(names) ^ set(joint_names)}")
        order = [names.index(n) for n in joint_names]
        self.joint_pos = torch.as_tensor(data["joint_pos"][:, order].copy(), device=device)
        self.pos = torch.as_tensor(data["base_pos_w"].copy(), device=device)
        # Official InstinctLab data is WXYZ; Lab 3.0 math and physics are XYZW.
        self.quat = torch.as_tensor(data["base_quat_w"][:, [1, 2, 3, 0]].copy(), device=device)
        norms = self.quat.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=0.002):
            raise ValueError("Invalid quaternion norms")
        self.quat = self.quat / norms[:, None]

        def diff(x):
            return torch.cat(((x[1:] - x[:-1]) * self.fps, torch.zeros_like(x[:1])), dim=0)

        joint_vel = diff(self.joint_pos)
        lin = diff(self.pos)
        dq = quat_mul(self.quat[1:], quat_conjugate(self.quat[:-1]))
        ang = torch.cat((axis_angle_from_quat(dq) * self.fps, torch.zeros_like(self.pos[:1])), dim=0)
        gravity = torch.zeros_like(self.pos)
        gravity[:, 2] = -1
        self.features = {
            "projected_gravity": quat_apply_inverse(self.quat, gravity),
            "joint_pos_rel": self.joint_pos - default_joint_pos[0],
            "joint_vel": joint_vel * 0.05,
            "base_lin_vel": quat_apply_inverse(self.quat, lin),
            "base_ang_vel": quat_apply_inverse(self.quat, ang),
        }
        if not all(torch.isfinite(v).all() for v in self.features.values()):
            raise ValueError("Nonfinite motion data")
        if self.filter_discontinuities:
            # Legacy task: exclude windows crossing suspected concatenation seams.
            seam = (lin.norm(dim=-1) > 10.0) | (joint_vel.abs().amax(dim=-1) > 100.0)
            starts = torch.arange(self.frames - 10, device=device)
            self.starts = starts[~seam[starts[:, None] + torch.arange(10, device=device)].any(dim=1)]
        else:
            # Include every complete 10-frame window, including the final one.
            # No speed threshold, seam mask, clipping, or content-based rejection.
            self.starts = torch.arange(self.frames - 10 + 1, device=device)
        if len(self.starts) == 0:
            raise ValueError("No valid AMP windows")
        self.device = device
        self.default_joint_pos = default_joint_pos[0].clone()
        self.mirror_order = torch.tensor(
            [
                joint_names.index(name.replace("left_", "right_", 1))
                if name.startswith("left_")
                else joint_names.index(name.replace("right_", "left_", 1))
                if name.startswith("right_")
                else i
                for i, name in enumerate(joint_names)
            ],
            device=device,
        )
        self.mirror_sign = torch.tensor(
            [-1.0 if "roll" in name or "yaw" in name else 1.0 for name in joint_names], device=device
        )
        self._reference_step = None
        self.manifest = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frames": self.frames,
            "fps": self.fps,
            "joint_names": joint_names,
            "quaternion_source": "wxyz",
            "quaternion_runtime": "xyzw",
            "valid_windows": len(self.starts),
            "window_frames": 10,
            "symmetric_augmentation_probability": 0.5,
            "sampling": "continuous per-environment stream, reset history repeated; NPZ lacks original clip labels",
            "start_range": [0.0, 0.9] if self.filter_discontinuities else [0.0, 1.0],
            "filter_discontinuities": self.filter_discontinuities,
            "reference_stream_frames": "legacy valid-window starts" if self.filter_discontinuities else "all source frames",
        }

    def sample(self, n):
        start = self.starts[torch.randint(len(self.starts), (n,), device=self.device)]
        idx = start[:, None] + torch.arange(10, device=self.device)
        return torch.cat([v[idx].flatten(1) for v in self.features.values()], dim=-1)

    def mirror(self, features):
        """Sagittal reflection: polar vectors flip Y, axial vectors flip X/Z."""
        result = {key: value.clone() for key, value in features.items()}
        result["joint_pos_rel"] = (features["joint_pos_rel"] + self.default_joint_pos)[
            ..., self.mirror_order
        ] * self.mirror_sign - self.default_joint_pos
        result["joint_vel"] = features["joint_vel"][..., self.mirror_order] * self.mirror_sign
        result["projected_gravity"][..., 1] *= -1
        result["base_lin_vel"][..., 1] *= -1
        result["base_ang_vel"][..., [0, 2]] *= -1
        return result

    def advance(self, n, step, reset_mask):
        """Keep the original per-environment reference history and 50% mirroring.

        The legacy task treats detected discontinuities as boundaries. The
        unfiltered task crosses every internal transition and resets only on
        episode reset or the actual end of the source file.
        """
        # Observation reads must be idempotent, including the step of an episode
        # reset. episode_length_buf remains zero for all reads on that step.
        if self._reference_step == step:
            return self._reference_output
        if not hasattr(self, "_cursor") or len(self._cursor) != n:
            self._cursor = torch.zeros(n, dtype=torch.long, device=self.device)
            self._mirror_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._reference_history = {
                k: torch.zeros(n, 10, v.shape[-1], device=self.device) for k, v in self.features.items()
            }
            reset_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        else:
            self._cursor += 1
        if self.filter_discontinuities:
            valid = torch.zeros(self.frames, dtype=torch.bool, device=self.device)
            valid[self.starts] = True
            reset_mask = reset_mask | ~valid[self._cursor.clamp(max=self.frames - 1)]
            eligible = self.starts[self.starts < int(self.frames * 0.9)]
        else:
            reset_mask = reset_mask | (self._cursor >= self.frames)
            eligible = torch.arange(self.frames, device=self.device)
        count = int(reset_mask.sum())
        self._cursor[reset_mask] = eligible[torch.randint(len(eligible), (count,), device=self.device)]
        self._mirror_mask[reset_mask] = torch.rand(count, device=self.device) < 0.5
        features = {k: v[self._cursor].clone() for k, v in self.features.items()}
        reflected = self.mirror(features)
        for key, value in features.items():
            value[self._mirror_mask] = reflected[key][self._mirror_mask]
            history = self._reference_history[key].roll(-1, dims=1)
            history[:, -1] = value
            history[reset_mask] = value[reset_mask, None]
            self._reference_history[key] = history
        self._reference_output = torch.cat([v.flatten(1) for v in self._reference_history.values()], dim=-1)
        self._reference_step = step
        return self._reference_output

    def reset(self):
        """Invalidate the reference stream for an explicit full environment reset."""
        self._reference_step = None
        if hasattr(self, "_cursor"):
            del self._cursor
