# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import functools, copy
import numpy as np
import trimesh
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh


def height_field_to_mesh(func: Callable) -> Callable:
    """Decorator to convert a height field function to a mesh function.

    This decorator converts a height field function to a mesh function by sampling the heights
    at a specified resolution and performing interpolation to obtain the intermediate heights.
    Additionally, it adds a border around the terrain to avoid artifacts at the edges.

    Args:
        func: The height field function to convert. The function should return a 2D numpy array
            with the heights of the terrain.

    Returns:
        The mesh function. The mesh function returns a tuple containing a list of ``trimesh``
        mesh objects and the origin of the terrain.
    """

    @functools.wraps(func)
    def wrapper(difficulty: float, cfg: HfTerrainBaseCfg):
        # check valid border width
        if cfg.border_width > 0 and cfg.border_width < cfg.horizontal_scale:
            raise ValueError(
                f"The border width ({cfg.border_width}) must be greater than or equal to the"
                f" horizontal scale ({cfg.horizontal_scale})."
            )
        # allocate buffer for height field (with border)
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
        border_pixels = int(cfg.border_width / cfg.horizontal_scale) + 1
        heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)
        # override size of the terrain to account for the border
        sub_terrain_size = [width_pixels - 2 * border_pixels, length_pixels - 2 * border_pixels]
        sub_terrain_size = [dim * cfg.horizontal_scale for dim in sub_terrain_size]
        # update the config
        terrain_size = copy.deepcopy(cfg.size)
        cfg.size = tuple(sub_terrain_size)
        # generate the height field
        z_gen = func(difficulty, cfg)
        # handle the border for the terrain
        heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = z_gen
        # set terrain size back to config
        cfg.size = terrain_size

        # convert to trimesh
        vertices, triangles = convert_height_field_to_mesh(
            heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # compute origin
        x1 = int((cfg.size[0] * 0.5 - 1) / cfg.horizontal_scale)
        x2 = int((cfg.size[0] * 0.5 + 1) / cfg.horizontal_scale)
        y1 = int((cfg.size[1] * 0.5 - 1) / cfg.horizontal_scale)
        y2 = int((cfg.size[1] * 0.5 + 1) / cfg.horizontal_scale)
        origin_z = np.max(heights[x1:x2, y1:y2]) * cfg.vertical_scale
        origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_z])
        # return mesh and origin
        return [mesh], origin

    return wrapper
