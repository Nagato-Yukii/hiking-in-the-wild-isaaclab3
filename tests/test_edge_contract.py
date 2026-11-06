"""A sharp-edge cylinder must yield finite penetration, including its centerline."""

import unittest
import numpy as np
import torch
import trimesh
from hikinglab3.cylinder import CylinderSpatialGrid
from hikinglab3.terrain_importer import prepare_terrain_mesh


class EdgeContract(unittest.TestCase):
    def test_disconnected_face_vertices_hide_geometric_edges(self):
        box = trimesh.creation.box()
        # Same surface, but every triangle owns separate vertices (a mesh join seam).
        raw = trimesh.Trimesh(vertices=box.vertices[box.faces].reshape(-1, 3),
                              faces=np.arange(len(box.faces)*3).reshape(-1, 3), process=False)
        self.assertEqual(len(raw.face_adjacency), 0)
        clean = prepare_terrain_mesh(raw)
        self.assertEqual(int((clean.face_adjacency_angles > np.deg2rad(70)).sum()), 12)
        np.testing.assert_allclose(clean.bounds, raw.bounds)
        self.assertEqual(len(raw.vertices), 36)  # Do not mutate the caller's mesh.
        again = prepare_terrain_mesh(clean)
        self.assertEqual(len(clean.vertices), len(again.vertices))
        self.assertEqual(len(clean.faces), len(again.faces))

    def test_penetration_and_axis_singularity(self):
        grid = CylinderSpatialGrid(
            np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.05]], np.float32), num_grid_cells=64, device="cpu"
        )
        points = torch.tensor([[0.5, 0.03, 0.0], [0.5, 0.1, 0.0], [0.5, 0.0, 0.0]])
        offsets = grid.get_points_penetration_offset(points)
        self.assertTrue(torch.isfinite(offsets).all())
        torch.testing.assert_close(offsets[0], torch.tensor([0.0, -0.02, 0.0]), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(offsets[1], torch.zeros(3))
        self.assertAlmostEqual(float(offsets[2].norm()), 0.05, places=5)
        self.assertAlmostEqual(float(offsets[2, 0]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
