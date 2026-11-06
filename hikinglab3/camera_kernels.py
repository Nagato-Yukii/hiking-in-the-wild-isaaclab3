# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted for the Hiking camera minimum ray-hit distance.
import warp as wp


@wp.kernel(enable_backward=False)
def raycast_dynamic_meshes_kernel(
    env_mask: wp.array(dtype=wp.bool),
    mesh: wp.array2d(dtype=wp.uint64),
    ray_starts: wp.array2d(dtype=wp.vec3),
    ray_directions: wp.array2d(dtype=wp.vec3),
    ray_hits: wp.array2d(dtype=wp.vec3),
    ray_distance: wp.array2d(dtype=wp.float32),
    ray_normal: wp.array2d(dtype=wp.vec3),
    ray_face_id: wp.array2d(dtype=wp.int32),
    ray_mesh_id: wp.array2d(dtype=wp.int16),
    mesh_positions: wp.array2d(dtype=wp.vec3),
    mesh_rotations: wp.array2d(dtype=wp.quat),
    max_dist: float = 1e6,
    return_normal: int = False,
    return_face_id: int = False,
    return_mesh_id: int = False,
    min_dist: float = 0.1,
):
    """Performs ray-casting against multiple dynamic meshes.

    This function performs ray-casting against the given meshes using the provided ray start positions
    and directions. The resulting ray hit positions are stored in the :obj:`ray_hits` array.

    The function utilizes the ``mesh_query_ray`` method from the ``wp`` module to perform the actual ray-casting
    operation. The maximum ray-cast distance is set to ``1e6`` units.

    Note:
        That the ``ray_starts``, ``ray_directions``, and ``ray_hits`` arrays should have compatible shapes
        and data types to ensure proper execution. Additionally, they all must be in the same frame.

        All arguments are expected to be batched with the first dimension (B, batch) being the number of envs
        and the second dimension (N, num_rays) being the number of rays. For Meshes, W is the number of meshes.

    .. warning::
        **Known race condition:** When two meshes are equidistant to the same ray, the
        ``atomic_min`` + equality-check pattern used for closest-hit resolution is not fully
        thread-safe.  Two threads may both pass the equality check and write different output
        fields (e.g., ``ray_hits`` from mesh A, ``ray_normal`` from mesh B).  In practice this
        is rare (requires exact floating-point tie) and the position output is still correct,
        but normals, face IDs, and mesh IDs may be inconsistent for the affected ray.
        See `warp#1058 <https://github.com/NVIDIA/warp/issues/1058>`_ for progress on a
        thread-safe fix.

    Args:
        env_mask: Boolean mask selecting which environments to process. Shape is (B,).
        mesh: The input mesh. The ray-casting is performed against this mesh on the device specified by the
            `mesh`'s `device` attribute.
        ray_starts: The input ray start positions. Shape is (B, N, 3).
        ray_directions: The input ray directions. Shape is (B, N, 3).
        ray_hits: The output ray hit positions. Shape is (B, N, 3).
        ray_distance: The closest hit distance buffer. Shape is (B, N). Updated via ``atomic_min`` for every
            thread that records a hit; used to resolve closest-hit among multiple meshes.
        ray_normal: The output ray hit normals. Shape is (B, N, 3), if ``return_normal`` is True. Otherwise,
            this array is not used.
        ray_face_id: The output ray hit face ids. Shape is (B, N,), if ``return_face_id`` is True. Otherwise,
            this array is not used.
        ray_mesh_id: The output ray hit mesh ids. Shape is (B, N,), if ``return_mesh_id`` is True. Otherwise,
            this array is not used.
        mesh_positions: The input mesh positions in world frame. Shape is (B, W, 3).
        mesh_rotations: The input mesh rotations in world frame. Shape is (B, W, 4).
        max_dist: The maximum ray-cast distance. Defaults to 1e6.
        return_normal: Whether to return the ray hit normals. Defaults to False.
        return_face_id: Whether to return the ray hit face ids. Defaults to False.
        return_mesh_id: Whether to return the mesh id. Defaults to False.
    """
    tid_mesh_id, tid_env, tid_ray = wp.tid()
    if not env_mask[tid_env]:
        return

    mesh_pose = wp.transform(mesh_positions[tid_env, tid_mesh_id], mesh_rotations[tid_env, tid_mesh_id])
    mesh_pose_inv = wp.transform_inverse(mesh_pose)
    direction = wp.transform_vector(mesh_pose_inv, ray_directions[tid_env, tid_ray])
    start_pos = wp.transform_point(mesh_pose_inv, ray_starts[tid_env, tid_ray])

    # ray cast against the mesh and store the hit position
    mesh_query_ray_t = wp.mesh_query_ray(mesh[tid_env, tid_mesh_id], start_pos, direction, max_dist)
    # if the ray hit, store the hit data
    if mesh_query_ray_t.result and mesh_query_ray_t.t > min_dist:
        wp.atomic_min(ray_distance, tid_env, tid_ray, mesh_query_ray_t.t)
        # TODO(warp#1058): Use the return value of atomic_min to avoid the non-thread-safe
        # equality check below. Currently warp atomic_min returns wrong values on GPU, so we
        # fall back to a racy read-back. When two meshes tie on distance, normals/face-ids/
        # mesh-ids may be written by different threads. The hit *position* is still correct
        # because all tying threads compute the same world-space point.
        if mesh_query_ray_t.t == ray_distance[tid_env, tid_ray]:
            # convert back to world space and update the hit data
            hit_pos = start_pos + mesh_query_ray_t.t * direction
            ray_hits[tid_env, tid_ray] = wp.transform_point(mesh_pose, hit_pos)

            # update the normal and face id if requested
            if return_normal == 1:
                n = wp.transform_vector(mesh_pose, mesh_query_ray_t.normal)
                ray_normal[tid_env, tid_ray] = n
            if return_face_id == 1:
                ray_face_id[tid_env, tid_ray] = mesh_query_ray_t.face
            if return_mesh_id == 1:
                ray_mesh_id[tid_env, tid_ray] = wp.int16(tid_mesh_id)
