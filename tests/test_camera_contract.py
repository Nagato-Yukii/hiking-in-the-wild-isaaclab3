"""Near robot geometry must not hide a valid farther terrain hit."""

import unittest
import numpy as np
import warp as wp
from hikinglab3.camera_kernels import raycast_dynamic_meshes_kernel


class CameraContract(unittest.TestCase):
    def test_minimum_hit_distance(self):
        meshes = []
        for x in [0.05, 1.0]:
            meshes.append(
                wp.Mesh(
                    points=wp.array([[x, -1.0, -1.0], [x, 1.0, -1.0], [x, 0.0, 1.0]], dtype=wp.vec3, device="cpu"),
                    indices=wp.array([0, 1, 2], dtype=wp.int32, device="cpu"),
                )
            )
        mask = wp.array([True], dtype=wp.bool, device="cpu")
        ids = wp.array([[m.id for m in meshes]], dtype=wp.uint64, device="cpu")
        start = wp.zeros((1, 1), dtype=wp.vec3, device="cpu")
        direction = wp.array([[[1.0, 0.0, 0.0]]], dtype=wp.vec3, device="cpu")
        positions = wp.zeros((1, 2), dtype=wp.vec3, device="cpu")
        rotations = wp.array([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]], dtype=wp.quat, device="cpu")
        for near, expected in [(0.1, 1.0), (0.0, 0.05)]:
            distance = wp.array([[np.inf]], dtype=wp.float32, device="cpu")
            wp.launch(
                raycast_dynamic_meshes_kernel,
                dim=(2, 1, 1),
                inputs=[
                    mask,
                    ids,
                    start,
                    direction,
                    wp.zeros((1, 1), dtype=wp.vec3, device="cpu"),
                    distance,
                    wp.zeros((1, 1), dtype=wp.vec3, device="cpu"),
                    wp.zeros((1, 1), dtype=wp.int32, device="cpu"),
                    wp.zeros((1, 1), dtype=wp.int16, device="cpu"),
                    positions,
                    rotations,
                    5.0,
                    0,
                    0,
                    0,
                    near,
                ],
                device="cpu",
            )
            self.assertAlmostEqual(float(distance.numpy()[0, 0]), expected, places=5)


if __name__ == "__main__":
    unittest.main()
