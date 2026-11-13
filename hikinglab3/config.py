# SPDX-License-Identifier: CC-BY-NC-4.0
"""Hiking in the Wild task on Isaac Lab 3.0/Newton; original G1 morphology."""

from pathlib import Path
import math
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, RayCasterCameraCfg, MultiMeshRayCasterCameraCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg
from .motors import beyondmimic_g1_29dof_actuators
from .terrain_importer import HikingTerrainImporter
from .terrain_cfg import ROUGH_TERRAINS_CFG
from .task_terms_cfg import *
from .observations import ObservationsCfg
from .camera import HikingDepthCameraCfg
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
URDF = (
    ROOT
    / "vendor/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
)

# Each link mesh is shared across environments; poses are tracked every frame.
G1_LINKS = [
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "waist_yaw_link",
    "waist_roll_link",
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]
VISUAL_TARGETS = [
    MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
        prim_expr="{ENV_REGEX_NS}/Robot/" + name, is_shared=True, track_mesh_transforms=True
    )
    for name in G1_LINKS
]


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        class_type=HikingTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        visual_material=sim.PreviewSurfaceCfg(diffuse_color=(0.22, 0.27, 0.30)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim.UsdFileCfg(
            usd_path=(ROOT / "assets/robot_path.txt").read_text().strip(),
            activate_contact_sensors=True,
            articulation_props=[NewtonArticulationCfg(self_collision_enabled=True)],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.9),
            joint_pos={
                ".*_hip_pitch_joint": -0.312,
                ".*_knee_joint": 0.669,
                ".*_ankle_pitch_joint": -0.363,
                ".*_elbow_joint": 0.6,
                "left_shoulder_roll_joint": 0.2,
                "right_shoulder_roll_joint": -0.2,
                ".*_shoulder_pitch_joint": 0.2,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators=beyondmimic_g1_29dof_actuators,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    left_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )
    right_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.04, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.12, size=[0.12, 0.0]),
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )
    camera = HikingDepthCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        mesh_prim_paths=["/World/ground"] + VISUAL_TARGETS,
        ray_alignment="yaw",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),
            width=64,
            height=36,
        ),
        data_types=["distance_to_image_plane"],
        update_period=0.02,
        max_distance=2.5,
        depth_clipping_behavior="max",
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.0487988662332928, 0.01, 0.4378029937970051),
            rot=(0.004363309284746571, 0.4067366430758002, 0.0, 0.9135367613482678),
            convention="world",
        ),
    )


@configclass
class HikingEnvCfg(ManagerBasedRLEnvCfg):
    amp_filter_discontinuities: bool = True
    scene = SceneCfg(num_envs=4096, env_spacing=2.5)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = G1Rewards()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()
    sim = SimulationCfg(dt=0.005, render_interval=4, physics=RoughPhysicsCfg().newton_mjwarp)
    seed = 42
    decimation = 4
    episode_length_s = 20.0

    def __post_init__(self):
        self.scene.terrain.terrain_generator.use_cache = True
        self.sim.physics.solver_cfg.njmax = 1000
        self.sim.physics.solver_cfg.nconmax = 300
        self.rewards.volume_points_penetration.params = {}
        self.rewards.feet_at_plane.params["height_offset"] = 0.058


@configclass
class HikingUnfilteredAmpEnvCfg(HikingEnvCfg):
    """Same locomotion task with all AMP reference frames and transitions retained."""

    amp_filter_discontinuities: bool = False
