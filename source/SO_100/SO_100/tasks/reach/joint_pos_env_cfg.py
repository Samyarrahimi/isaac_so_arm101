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

import os
os.environ["HYDRA_FULL_ERROR"] = "1"  # for better error messages during development

# import mdp
import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass
from SO_100.tasks.reach.robots import SO_ARM100_CAMERA_CFG, SO_ARM100_ROSCON_CFG
from SO_100.tasks.reach.reach_env_cfg import ReachEnvCfg
##
# Scene definition
##


@configclass
class SoArm100ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot
        self.scene.robot = SO_ARM100_CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper"]

        # TODO: reorient command target

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.5,
            preserve_order=True,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["gripper"]
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

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
class SoArm100ReachEnvCfg_PLAY(SoArm100ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# ----------------------------------------------------------------
# --------------- ROSCON ES 2025 asset ---------------------------
# ----------------------------------------------------------------


@configclass
class SoArm100ReachRosConEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = SO_ARM100_ROSCON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["wrist_2_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["wrist_2_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["wrist_2_link"]

        self.rewards.end_effector_orientation_tracking.weight = 0.0

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale=0.5,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = ["wrist_2_link"]
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class SoArm100ReachRosConEnvCfg_PLAY(SoArm100ReachRosConEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
