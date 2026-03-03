# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import os
os.environ["HYDRA_FULL_ERROR"] = "1"  # for better error messages during development

# import mdp
import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG

from SO_100.tasks.move_object.robots import SO_ARM100_CAMERA_CFG
from SO_100.tasks.move_object.move_env_cfg import (
    MoveEnvCfg,
    EventCfgFineTune,
    MULTI_OBJECT_ASSETS_CFG,
    BOLT_OBJECT_CFG,
    GEAR_OBJECT_CFG,
    NUT_OBJECT_CFG,
    PEG_OBJECT_CFG,
    BOX_OBJECT_CFG,
)

##
# Camera configs (shared by all PLAY variants; not added during training)
##

_CONTEXT_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/context_camera",
    update_period=0.04,
    height=360,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=5.75,
        focus_distance=700.0,
        horizontal_aperture=5.76,
        vertical_aperture=3.24,
        clipping_range=(0.0001, 1.0e5),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(1, -1.3, 1.5),
        rot=(0.8189, 0.3664, 0.1354, 0.4202),
        convention="opengl",
    ),
    update_latest_camera_pose=True,
)

_WRIST_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/wrist/Realsense/RSD455/wrist_camera",
    update_period=0.04,
    height=360,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.93,
        focus_distance=0.5,
        horizontal_aperture=3.896,
        vertical_aperture=2.453,
        clipping_range=(0.0001, 1.0e5),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0115, 0.0),
        rot=(0.5, 0.5, 0.5, 0.5),
        convention="opengl",
    ),
)

##
# Scene definition
##

@configclass
class SoArm100MoveObjectEnvCfg(MoveEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = SO_ARM100_CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.5,
            preserve_order=True,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 1.5},
            close_command_expr={"gripper": -0.1},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["gripper"]

        # Set DexCube as object for pretraining
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.015)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        self.scene.contact_sensor_moving = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/jaw",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        self.scene.contact_sensor_fixed = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, -0.09, 0.01],
                    ),
                ),
            ],
        )

        # cube_marker_cfg = FRAME_MARKER_CFG.copy()
        # cube_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        # cube_marker_cfg.prim_path = "/Visuals/CubeFrameMarker"
        # the actual prim path should be something under Object (e.g. bolt, nut, etc.)
        # but this is only for visualization and for training so we can comment it during playing policy
        # self.scene.cube_marker = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     visualizer_cfg=cube_marker_cfg,
        #     debug_vis=False,  # disable visualization
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Object",
        #             name="cube",
        #             offset=OffsetCfg(
        #                 pos=(0.0, 0.0, 0.0),
        #             ),
        #         ),
        #     ],
        # )

        # mesh for context camera (not a necessity but just to make it nicer)
        # rotation here (0.8601, -0.4253, -0.1244, 0.2535) is not the same as rotation in context_camera but they look similar
        # position is absolutely correct but the orientation is a bit off. at the end it does not matter as we will not use the camera in Camera_SG2_OX03CC_5200_GMSL2_H60YA.usd for observations, we will use it just for visualization. The important thing is that the camera pose is correct, which it is.
        # this just acts as a mesh which is also not a necessity)
        # self.scene.context_camera_mesh = AssetBaseCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/base/context_camera_mesh",
        #     init_state=AssetBaseCfg.InitialStateCfg(
        #         pos=(0.8, -1.5, 1.7),
        #         rot=(0.8601, -0.4253, -0.1244, 0.2535),
        #     ),
        #     spawn=UsdFileCfg(
        #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Sensors/Sensing/SG2/H60YA/Camera_SG2_OX03CC_5200_GMSL2_H60YA.usd",
        #         scale=(1.0, 1.0, 1.0),
        #     ),
        # )


@configclass
class SoArm100MoveObjectEnvCfg_PLAY(SoArm100MoveObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # Enable cameras only for play (not needed during training)
        self.scene.context_camera = _CONTEXT_CAMERA_CFG
        self.scene.wrist_camera = _WRIST_CAMERA_CFG


# ---------------------------------------------------------------------------
# Fine-tuning env: all objects, scale diversity, per-reset colour randomization
# ---------------------------------------------------------------------------

@configclass
class SoArm100MoveObjectFinetuneEnvCfg(SoArm100MoveObjectEnvCfg):
    """Fine-tuning env that randomises object shape, scale and colour.

    Object assignment strategy
    --------------------------
    ``MultiAssetSpawnerCfg`` with ``random_choice=False`` assigns assets
    round-robin across envs at sim startup (env_i gets pool[i % 15]).
    With 4 096 envs → ~273 envs per variant; with 50 play envs → ~3 per
    variant.  Distribution is always proportional regardless of num_envs.

    Colour randomisation
    --------------------
    ``randomize_visual_color`` is applied at every episode reset
    (``EventCfgFineTune``).  It requires ``replicate_physics=False`` so that
    each env's material can be mutated independently.

    Scale diversity
    ---------------
    Handled statically through the pool (3 scales per object); PhysX does
    not support runtime collision-mesh rescaling.
    """

    def __post_init__(self):
        super().__post_init__()

        # Replace the single-object spawn with the 15-entry multi-asset pool.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=MultiAssetSpawnerCfg(
                assets_cfg=MULTI_OBJECT_ASSETS_CFG,
                random_choice=False,  # round-robin → guaranteed proportional coverage
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.015)),
        )
        # Required for per-env material mutation in randomize_visual_color.
        self.scene.replicate_physics = False
        # Swap in the extended event config that adds per-reset colour randomization.
        self.events = EventCfgFineTune()


@configclass
class SoArm100MoveObjectFinetuneEnvCfg_PLAY(SoArm100MoveObjectFinetuneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # Enable cameras only for play (not needed during training)
        self.scene.context_camera = _CONTEXT_CAMERA_CFG
        self.scene.wrist_camera = _WRIST_CAMERA_CFG


# ---------------------------------------------------------------------------
# Per-object fine-tuning envs: one specialised policy per object type.
# Each env inherits the full robot/sensor/action setup from
# SoArm100MoveObjectEnvCfg and only replaces the spawned object.
# Use these after pretraining on the multi-object env to fine-tune a
# dedicated policy for a single object type.
# ---------------------------------------------------------------------------

@configclass
class SoArm100MoveBoltFinetuneEnvCfg(SoArm100MoveObjectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object = BOLT_OBJECT_CFG
        # replicate_physics=True (default): all envs spawn the same bolt →
        # PhysX solves physics once and tiles, which is faster and avoids
        # totalAggregatePairsCapacity overflow.
        # Colour randomisation is deferred to the PLAY variant where cameras
        # are active and replicate_physics=False is required.


@configclass
class SoArm100MoveBoltFinetuneEnvCfg_PLAY(SoArm100MoveBoltFinetuneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.replicate_physics = False
        self.events = EventCfgFineTune()
        self.scene.context_camera = _CONTEXT_CAMERA_CFG
        self.scene.wrist_camera = _WRIST_CAMERA_CFG


@configclass
class SoArm100MoveGearFinetuneEnvCfg(SoArm100MoveObjectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object = GEAR_OBJECT_CFG


@configclass
class SoArm100MoveGearFinetuneEnvCfg_PLAY(SoArm100MoveGearFinetuneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.replicate_physics = False
        self.events = EventCfgFineTune()
        self.scene.context_camera = _CONTEXT_CAMERA_CFG
        self.scene.wrist_camera = _WRIST_CAMERA_CFG


@configclass
class SoArm100MoveNutFinetuneEnvCfg(SoArm100MoveObjectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object = NUT_OBJECT_CFG


@configclass
class SoArm100MoveNutFinetuneEnvCfg_PLAY(SoArm100MoveNutFinetuneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.replicate_physics = False
        self.events = EventCfgFineTune()
        self.scene.context_camera = _CONTEXT_CAMERA_CFG
        self.scene.wrist_camera = _WRIST_CAMERA_CFG


@configclass
class SoArm100MovePegFinetuneEnvCfg(SoArm100MoveObjectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object = PEG_OBJECT_CFG


@configclass
class SoArm100MovePegFinetuneEnvCfg_PLAY(SoArm100MovePegFinetuneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.replicate_physics = False
        self.events = EventCfgFineTune()
        self.scene.context_camera = _CONTEXT_CAMERA_CFG
        self.scene.wrist_camera = _WRIST_CAMERA_CFG


@configclass
class SoArm100MoveBoxFinetuneEnvCfg(SoArm100MoveObjectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object = BOX_OBJECT_CFG


@configclass
class SoArm100MoveBoxFinetuneEnvCfg_PLAY(SoArm100MoveBoxFinetuneEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.replicate_physics = False
        self.events = EventCfgFineTune()
        self.scene.context_camera = _CONTEXT_CAMERA_CFG
        self.scene.wrist_camera = _WRIST_CAMERA_CFG
