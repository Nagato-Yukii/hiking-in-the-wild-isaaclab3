# Derived from Project-Instinct / InstinctLab (CC-BY-NC-4.0).
# Isaac Lab 3.0 migration; see THIRD_PARTY.md for provenance.
from __future__ import annotations
import numpy as np
import torch
import random
import trimesh
from .cylinder import CylinderSpatialGrid


class EdgeCylinder:
    """Base class for edge detectors."""

    def __init__(self, cfg: EdgeCylinderCfg):
        self.cfg: EdgeCylinderCfg = cfg
        self.angle_threshold = cfg.angle_threshold

    def generate(self, mesh: trimesh.Trimesh, device="cpu") -> None:
        """Detect sharp edges in the mesh and store the edge cylinder as virtual obstacle.

        Args:
            mesh: The trimesh object to analyze.

        Returns:
            A np array batch indicating the edges: (num_edges, 6)
            - x, y, z coordinates of the edge start point
            - x, y, z coordinates of the edge end point
        """
        angles = mesh.face_adjacency_angles
        # convert max_angle in degrees to radians
        threshold = np.deg2rad(self.angle_threshold)
        # pick only those adjacencies whose angle exceeds threshold
        sharp_mask = angles > threshold
        if not np.any(sharp_mask):
            edge_end_points = np.empty((0, 6), dtype=np.float32)
            print("[WARNING] No sharp edges detected.")
        else:
            # get the corresponding edges (vertex index pairs)
            # face_adjacency_edges is (n_adj, 2) vertex indices for each adjacency
            sharp_edges = mesh.face_adjacency_edges[sharp_mask]

            # look up vertex coordinates
            v = mesh.vertices
            # build (num_edges, 6) array: [x0,y0,z0, x1,y1,z1]
            # build the (num_edges, 6) array of sharp edge end‐point coordinates
            edge_coords = np.hstack([v[sharp_edges[:, 0]], v[sharp_edges[:, 1]]])
            edge_end_points = self.process_edges(edge_coords)
            print(f"Detected {edge_end_points.shape[0]} edges after processing.")
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.edges_pyt = torch.tensor(edge_end_points, dtype=torch.float32, device=self.device)
        # create a cylinder spatial grid for the edges and for penetration offset computation
        if edge_end_points.size > 0:
            self.cylinders = CylinderSpatialGrid(
                cylinders=np.concatenate(
                    [
                        edge_end_points,
                        np.ones_like(edge_end_points[:, :1]) * self.cfg.cylinder_radius,
                    ],
                    axis=1,
                ),
                num_grid_cells=self.cfg.num_grid_cells,
                device=self.device,
            )
        else:
            self.cylinders = None

    def disable_visualizer(self):
        if hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer.set_visibility(False)

    def visualize(self):
        if self.edges_pyt.numel() == 0:
            return

        if not hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer = VisualizationMarkers(self.cfg.visualizer)
            self._cylinder_rotate_y_90 = math_utils.quat_from_angle_axis(
                angle=torch.tensor([np.pi / 2], device=self.device),
                axis=torch.tensor([[0.0, 1.0, 0.0]], device=self.device),
            )  # shape (1, 4)

        trans = (self.edges_pyt[:, :3] + self.edges_pyt[:, 3:6]) / 2
        # compute the direction quaternion
        direction = self.edges_pyt[:, 3:6] - self.edges_pyt[:, :3]
        default_direction = torch.zeros_like(direction)
        default_direction[:, 0] = 1.0
        normalized_direction = direction / torch.norm(direction, dim=-1, keepdim=True)  # arrow-direction
        axis = torch.cross(default_direction, normalized_direction, dim=-1)
        dot_prod_ = torch.sum(default_direction * normalized_direction, dim=-1)
        angle = torch.acos(torch.clamp(dot_prod_, -1.0, 1.0))
        quat = math_utils.quat_from_angle_axis(
            angle,
            axis,
        )
        quat = math_utils.quat_mul(quat, self._cylinder_rotate_y_90.expand(quat.shape[0], -1))
        # compute the scale to match the length and the edge thickness.
        scales = torch.ones(len(self.edges_pyt), 3, device=self.device)
        scales[:, 0] = self.cfg.cylinder_radius
        scales[:, 1] = self.cfg.cylinder_radius
        scales[:, 2] = torch.norm(direction, dim=-1)
        self._cylinder_visualizer.visualize(
            translations=trans,
            orientations=quat,
            scales=scales,
        )
        self._cylinder_visualizer.set_visibility(True)

    def get_points_penetration_offset(self, points):
        return (
            self.cylinders.get_points_penetration_offset(points)
            if self.cylinders is not None
            else torch.zeros_like(points, device=self.device)
        )

    def process_edges(self, edge_coords: np.ndarray) -> np.ndarray:
        """Process the edge coordinates.

        Args:
            edge_coords: The edge coordinates array of shape (num_edges, 6).

        Returns:
            A np array of processed edge coordinates.
        """
        return edge_coords


class GreedyconcatEdgeCylinder(EdgeCylinder):
    """Detects sharp edges in a mesh using the Laplacian operator."""

    def __init__(self, cfg: GreedyconcatEdgeCylinderCfg):
        self.cfg: GreedyconcatEdgeCylinderCfg = cfg
        self.angle_threshold = cfg.angle_threshold

    def process_edges(self, edge_coords: np.ndarray) -> np.ndarray:
        line_pts = edge_coords.reshape(-1, 3)
        V, inv_idx = np.unique(line_pts, axis=0, return_inverse=True)
        E_pairs = inv_idx.reshape(-1, 2)

        adj_list = {i: set() for i in range(V.shape[0])}
        for u, v in E_pairs:
            if u != v:
                adj_list[u].add(v)
                adj_list[v].add(u)

        num_edges_V = np.array([len(adj_list[i]) for i in range(V.shape[0])], dtype=int)
        available_edges = set(np.where(num_edges_V > 0)[0])

        cos_threshold = np.cos(np.deg2rad(self.cfg.adjacent_angle_threshold))

        processed_edge_coords = []

        def compute_max_distance_to_line_vec(V, v_set):
            A = V[v_set[0]]
            B = V[v_set[-1]]
            AB = B - A
            norm_AB = np.linalg.norm(AB)
            if norm_AB == 0:
                return v_set[0], 0.0
            pts = V[v_set]
            dists = np.linalg.norm(np.cross(pts - A, pts - B), axis=1) / norm_AB
            max_idx = np.argmax(dists)
            return v_set[max_idx], dists[max_idx]

        while available_edges:
            selected_vertex = random.choice(list(available_edges))
            v_set = [selected_vertex]

            if adj_list[selected_vertex]:
                neighbor = next(iter(adj_list[selected_vertex]))
                v_set.append(neighbor)
                adj_list[selected_vertex].remove(neighbor)
                adj_list[neighbor].remove(selected_vertex)
                for vid in [selected_vertex, neighbor]:
                    num_edges_V[vid] -= 1
                    if num_edges_V[vid] == 0:
                        available_edges.discard(vid)

            while True:
                find_neighbor = False
                start, end = v_set[0], v_set[-1]

                neighbors = list(adj_list[start])
                if neighbors:
                    dirs = V[start] - V[neighbors]
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    start_dir = V[v_set[1]] - V[start]
                    start_dir /= np.linalg.norm(start_dir)
                    dots = dirs @ start_dir
                    idx = np.where(dots > cos_threshold)[0]
                    if idx.size > 0:
                        n = neighbors[idx[0]]
                        v_set.insert(0, n)
                        adj_list[start].remove(n)
                        adj_list[n].remove(start)
                        for vid in [start, n]:
                            num_edges_V[vid] -= 1
                            if num_edges_V[vid] == 0:
                                available_edges.discard(vid)
                        find_neighbor = True

                neighbors = list(adj_list[end])
                if neighbors:
                    dirs = V[neighbors] - V[end]
                    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                    end_dir = V[end] - V[v_set[-2]]
                    end_dir /= np.linalg.norm(end_dir)
                    dots = dirs @ end_dir
                    idx = np.where(dots > cos_threshold)[0]
                    if idx.size > 0:
                        n = neighbors[idx[0]]
                        v_set.append(n)
                        adj_list[end].remove(n)
                        adj_list[n].remove(end)
                        for vid in [end, n]:
                            num_edges_V[vid] -= 1
                            if num_edges_V[vid] == 0:
                                available_edges.discard(vid)
                        find_neighbor = True

                if not find_neighbor:
                    break

            while len(v_set) >= self.cfg.min_points:
                for i in range(len(v_set) - 1):
                    max_vi, max_dist = compute_max_distance_to_line_vec(V, v_set[i:])
                    if max_dist < 0.05:
                        break
                if len(v_set) - i >= self.cfg.min_points:
                    processed_edge_coords.append(np.concatenate([V[v_set[i]], V[v_set[-1]]]))
                v_set = v_set[: i + 1]

        return np.array(processed_edge_coords, dtype=np.float32)
