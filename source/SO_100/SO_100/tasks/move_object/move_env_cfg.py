# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import os
os.environ["HYDRA_FULL_ERROR"] = "1"

# from . import mdp
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from SO_100.tasks.move_object import mdp as my_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import randomize_visual_color
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
from isaaclab.utils import configclass
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from pathlib import Path

##
# Scene definition
##

TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"

# ---------------------------------------------------------------------------
# Helpers for multi-object fine-tuning pool
# ---------------------------------------------------------------------------

def _object_usd(usd_path: str, scale: float) -> sim_utils.UsdFileCfg:
    """UsdFileCfg for a graspable object. Rigid body is assigned at root by IsaacLab via rigid_props."""
    return sim_utils.UsdFileCfg(
        usd_path=usd_path,
        scale=(scale, scale, scale),
    )

# 15-entry round-robin pool: 5 objects × 3 scales (×0.8, ×1.0, ×1.2 of base).
# Move task base scales: factory objects = 1.5; green_block = 0.7.
MULTI_OBJECT_ASSETS_CFG: list = [
    # Bolt
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_bolt_m16.usd", 0.9),
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_bolt_m16.usd", 1.0),
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_bolt_m16.usd", 1.1),
    # Gear
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_gear_large.usd", 0.6),
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_gear_large.usd", 0.7),
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_gear_large.usd", 0.8),
    # Nut
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_nut_m16.usd", 0.9),
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_nut_m16.usd", 1.0),
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_nut_m16.usd", 1.1),
    # Peg
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_peg_8mm.usd", 2.0),
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_peg_8mm.usd", 2.1),
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_peg_8mm.usd", 2.2),
    # Green block
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/green_block.usd", 0.5),
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/green_block.usd", 0.6),
    _object_usd(f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/green_block.usd", 0.7),
]


# ---------------------------------------------------------------------------
# Per-object RigidObjectCfg constants.
# Defined at module level so they can be imported directly by env configs —
# @configclass moves class-level fields into dataclass field machinery,
# making them inaccessible as class attributes.
# ---------------------------------------------------------------------------

_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

BOLT_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_bolt_m16.usd",
        scale=(1.0, 1.0, 1.0),
        rigid_props=_RIGID_PROPS,
        articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.015)),
)

GEAR_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_gear_large.usd",
        scale=(0.6, 0.6, 0.8),
        rigid_props=_RIGID_PROPS,
        articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.015)),
)

NUT_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_nut_m16.usd",
        scale=(1.0, 1.0, 1.0),
        rigid_props=_RIGID_PROPS,
        articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.015)),
)

PEG_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/factory_peg_8mm.usd",
        scale=(2.5, 2.5, 2.5),
        rigid_props=_RIGID_PROPS,
        articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.015)),
)

BOX_OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Objects/green_block.usd",
        scale=(0.6, 0.6, 0.6),
        rigid_props=_RIGID_PROPS,
        articulation_props=ArticulationRootPropertiesCfg(articulation_enabled=False),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.015)),
)


@configclass
class MoveObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for the move scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    # warehouse = AssetBaseCfg(
    #     prim_path="/World/warehouse",
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd"),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, -1.05),
    #     ),
    # )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # context camera
    context_camera_mesh = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table/Context_Camera",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, -1.0, 1.3), rot=(0.8785, -0.36, -0.07, 0.3)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Sensors/Sensing/SG2/H60YA/Camera_SG2_OX03CC_5200_GMSL2_H60YA.usd",
            scale=(1.0, 1.0, 1.0)
        ),
    )

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING

    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    #cube_marker: FrameTransformerCfg = MISSING

    # contact sensor
    contact_sensor_moving: ContactSensorCfg = MISSING
    contact_sensor_fixed: ContactSensorCfg = MISSING

    # Cameras are None during training (not needed by policy observations).
    # PLAY configs enable them so play.py can save images.
    context_camera: TiledCameraCfg | None = None
    wrist_camera: TiledCameraCfg | None = None


@configclass
class MoveObjectSceneWithBoltCfg(MoveObjectSceneCfg):
    """Scene Class for Move Object Task with Bolt considered for kitting, sorting"""
    object = BOLT_OBJECT_CFG

@configclass
class MoveObjectSceneWithGearCfg(MoveObjectSceneCfg):
    """Scene Class for Move Object Task with Gear considered for kitting, sorting"""
    object = GEAR_OBJECT_CFG

@configclass
class MoveObjectSceneWithNutCfg(MoveObjectSceneCfg):
    """Scene Class for Move Object Task with Nut considered for kitting, sorting"""
    object = NUT_OBJECT_CFG

@configclass
class MoveObjectSceneWithPegCfg(MoveObjectSceneCfg):
    """Scene Class for Move Object Task with Peg considered for kitting, sorting"""
    object = PEG_OBJECT_CFG

@configclass
class MoveObjectSceneWithBoxCfg(MoveObjectSceneCfg):
    """Scene Class for Move Object Task with Box considered for stacking, unstacking scenarios"""
    object = BOX_OBJECT_CFG


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.3, -0.1),
            pos_z=(0.2, 0.35),
            roll=(0.0, 0.0), # x axis
            pitch=(0.0, 0.0), # y axis 180 degree = 3.14 rad
            yaw=(0.0, 0.0), # z axis
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)#, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)#, noise=Unoise(n_min=-0.01, n_max=0.01))
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    #reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_joints = EventTerm(
        func=my_mdp.randomize_robot_joint_positions,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "joint_noise_std": 0.05,
        },
    )

    random_shoulder_rotation = EventTerm(
        func=my_mdp.randomize_shoulder_rotation,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "min_angle": -1.56,
            "max_angle":  1.56,
        },
    )

    random_shoulder_pitch = EventTerm(
        func=my_mdp.randomize_shoulder_pitch,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "min_angle": 0.0,
            "max_angle":  3.48,
        },
    )

    reset_object_position = EventTerm(
        func=my_mdp.set_object_position,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
            "local_offset_xyz": (0.0, -0.09, 0.01),
            "extra_z_lower": 0.0,
        },
    )

    gripper_grasp_object = EventTerm(
        func = my_mdp.grasp_object,
        mode = "reset",
        params = {
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
            "gripper_closed_value": -0.1
        },
    )


@configclass
class EventCfgFineTune(EventCfg):
    """EventCfg with per-reset visual color randomization for domain randomization.

    Requires ``replicate_physics=False`` on the scene (set in the finetune env cfg).
    Colors are sampled uniformly from the full RGB cube on every episode reset so the
    policy cannot rely on object colour as a cue.
    """

    randomize_object_color: EventTerm = EventTerm(
        func=randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mesh_name": "/*/visuals",
            "colors": {"r": (0.05, 1.0), "g": (0.05, 1.0), "b": (0.05, 1.0)},
            "event_name": "randomize_object_color",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=1.0)

    grasp_object = RewTerm(func=my_mdp.check_grasped, weight=20.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    object_released = DoneTerm(func=my_mdp.check_released, params={"force_threshold": 0.0})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class MoveEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Moving environment."""

    # Scene settings
    scene: MoveObjectSceneCfg | MoveObjectSceneWithBoltCfg | MoveObjectSceneWithGearCfg | MoveObjectSceneWithNutCfg | MoveObjectSceneWithPegCfg | MoveObjectSceneWithBoxCfg = MoveObjectSceneWithBoxCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 8 * 2**15  # 262144; default 163840 overflows with gear mesh
        self.sim.physx.friction_correlation_distance = 0.00625
        #self.sim.physx.gpu_collision_stack_size = 2**29  # 512 MB; avoids collisionStackSize overflow with many envs
        