# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted for the Hiking camera minimum ray-hit distance.
import warp as wp
from isaaclab_newton.sensors import LegacyMultiMeshRayCasterCamera
from isaaclab.sensors import MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster import kernels as ray_caster_kernels
from isaaclab.utils import configclass
from . import camera_kernels


class HikingDepthCamera(LegacyMultiMeshRayCasterCamera):
    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        self._update_ray_infos(env_mask)

        # Increment frame count for updated environments
        self._update_frame(env_mask, frame_op=1)

        self._update_mesh_transforms()

        return_normal = "normals" in self.cfg.data_types

        # Fill ray hit, distance, and optional normal buffers with inf for masked environments.
        wp.launch(
            ray_caster_kernels.fill_ray_hits_distance_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, return_normal],
            outputs=[self.ray_hits_w.warp, self._ray_distance_cam_wp, self._ray_normal_w],
            device=self._device,
        )

        # Ray-cast against all meshes; closest hit wins via atomic_min on ray_distance.
        wp.launch(
            camera_kernels.raycast_dynamic_meshes_kernel,
            dim=(self._mesh_ids_wp.shape[1], self._num_envs, self.num_rays),
            inputs=[
                env_mask,
                self._mesh_ids_wp,
                self._ray_starts_w,
                self._ray_directions_w,
                self.ray_hits_w.warp,
                self._ray_distance_cam_wp,
                self._ray_normal_w,
                self._ray_face_id_wp,
                self._ray_mesh_id_wp,
                self._mesh_positions_w,
                self._mesh_orientations_w,
                float(ray_caster_kernels.CAMERA_RAYCAST_MAX_DIST),
                int(return_normal),
                int(False),  # return_face_id
                int(self.cfg.update_mesh_ids),
                float(self.cfg.min_distance),
            ],
            device=self._device,
        )

        if "distance_to_image_plane" in self.cfg.data_types:
            wp.launch(
                ray_caster_kernels.compute_distance_to_image_plane_to_image_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[
                    env_mask,
                    self._data.quat_w_world.warp,
                    self._ray_distance_cam_wp,
                    self._ray_directions_w,
                    int(self.image_shape[1]),
                    bool(self._depth_clip_enabled),
                    float(self.cfg.max_distance),
                    self._depth_clip_fill_value,
                ],
                outputs=[
                    self._data.output["distance_to_image_plane"].warp,
                ],
                device=self._device,
            )

        if "distance_to_camera" in self.cfg.data_types:
            wp.launch(
                ray_caster_kernels.copy_float2d_to_image1_depth_clipped_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[
                    env_mask,
                    self._ray_distance_cam_wp,
                    int(self.image_shape[1]),
                    bool(self._depth_clip_enabled),
                    float(self.cfg.max_distance),
                    self._depth_clip_fill_value,
                ],
                outputs=[
                    self._data.output["distance_to_camera"].warp,
                ],
                device=self._device,
            )

        if return_normal:
            wp.launch(
                ray_caster_kernels.copy_vec3_2d_to_image3_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, self._ray_normal_w, int(self.image_shape[1]), self._data.output["normals"].warp],
                device=self._device,
            )

        if self.cfg.update_mesh_ids:
            wp.launch(
                ray_caster_kernels.copy_int16_2d_to_image1_masked_kernel,
                dim=(self._num_envs, self.num_rays),
                inputs=[env_mask, self._ray_mesh_id_wp, int(self.image_shape[1]), self._data.image_mesh_ids.warp],
                device=self._device,
            )


@configclass
class HikingDepthCameraCfg(MultiMeshRayCasterCameraCfg):
    class_type = HikingDepthCamera
    min_distance: float = 0.1
