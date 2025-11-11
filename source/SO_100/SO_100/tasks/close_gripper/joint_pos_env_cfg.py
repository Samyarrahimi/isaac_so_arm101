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
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.utils import configclass
from SO_100.robots import SO_ARM100_CFG, SO_ARM100_CAMERA_CFG  # noqa: F401
from SO_100.tasks.close_gripper.close_gripper_env_cfg import CloseGripperEnvCfg

##
# Scene definition
##


# damping and stiffness values must be set to 0 for all joints other than Gripper
# otherwise, the joints change position during training eventhough no action is applied
# Gripper action is contnious and not Binary anymore
@configclass
class SoArm100CloseGripperEnvCfg(CloseGripperEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = SO_ARM100_CAMERA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        # self.actions.arm_action = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        #     scale=0.5,
        #     use_default_offset=True,
        # )
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["Gripper"],
        #     open_command_expr={"Gripper": 0.5},
        #     close_command_expr={"Gripper": 0.0},
        # )
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Gripper"],
            scale=0.5,
            use_default_offset=True,
            clip={"Gripper": (0.026, 0.698)},  # gripper joint limits
        )


@configclass
class SoArm100CloseGripperEnvCfg_PLAY(SoArm100CloseGripperEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
