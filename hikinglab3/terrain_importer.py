# SPDX-License-Identifier: CC-BY-NC-4.0
from isaaclab.terrains import TerrainImporter


def prepare_terrain_mesh(mesh):
    """Restore InstinctLab's topology cleanup before collision and edge detection."""
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


class HikingTerrainImporter(TerrainImporter):
    """Keep the generated mesh for the paper's edge detection mechanism."""

    def import_mesh(self, name, mesh):
        mesh = prepare_terrain_mesh(mesh)
        self.edge_mesh = mesh
        return super().import_mesh(name, mesh)
