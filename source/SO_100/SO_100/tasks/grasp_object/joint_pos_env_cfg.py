# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# import mdp
import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG

from SO_100.tasks.grasp_object.robots import SO_ARM100_CAMERA_CFG
from SO_100.tasks.grasp_object.grasp_object_env_cfg import GraspObjectEnvCfg

##
# Scene definition
##

@configclass
class SoArm100GraspObjectEnvCfg(GraspObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
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

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.015], rot=[1, 0, 0, 0]),
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

        cube_marker_cfg = FRAME_MARKER_CFG.copy()
        cube_marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        cube_marker_cfg.prim_path = "/Visuals/CubeFrameMarker"
        self.scene.cube_marker = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            visualizer_cfg=cube_marker_cfg,
            debug_vis=False,  # disable visualization
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    name="cube",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )

        self.scene.context_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/context_camera",
            update_period=0.04,
            height=360,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.0001, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1, -1.3, 1.5),
                rot=(0.8189, 0.3664, 0.1354, 0.4202),
                convention="opengl",
            ),
            update_latest_camera_pose=True,
        )

        self.scene.wrist_camera = TiledCameraCfg(
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
class SoArm100GraspObjectEnvCfg_PLAY(SoArm100GraspObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
