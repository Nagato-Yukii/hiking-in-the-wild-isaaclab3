# SPDX-License-Identifier: CC-BY-NC-4.0
from isaaclab.terrains import TerrainImporter


class HikingTerrainImporter(TerrainImporter):
    """Keep the generated mesh for the paper's edge detection mechanism."""

    def import_mesh(self, name, mesh):
        self.edge_mesh = mesh
        return super().import_mesh(name, mesh)
