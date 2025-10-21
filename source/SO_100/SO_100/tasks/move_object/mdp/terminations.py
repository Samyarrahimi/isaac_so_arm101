# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Additional imports from pxr and USD for transforms:
from pxr import Gf, Usd, UsdGeom


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold

def initialize_grasped_start(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | list[int] | None = None,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        gripper_closed_value: float = 0.0,
        object_offset_local: tuple[float, float, float] = (0.0, 0.0, -0.02),
        start_ee_height: float = 0.10
    ) -> torch.BoolTensor:
    """
    Initialize the episode start so that the object is already grasped by the robot.
    - Robot's gripper joint is set to `gripper_closed_value`.
    - Robot ther joints set to default.
    - Object pose is placed relative to the robot's fixed gripper with offset.
    - End-effector (gripper) is at `start_ee_height` above the table.
    
    Args:
        env: the ManagerBasedRLEnv instance.
        env_ids: indices of environments to apply this to (None = all).
        robot_cfg: SceneEntityCfg for the robot articulation.
        object_cfg: SceneEntityCfg for the object rigid body.
        gripper_closed_value: joint value representing closed gripper.
        object_offset_local: (x, y, z) offset of the object in the gripper local frame.
        start_ee_height: desired height above table for end-effector.
    
    Returns:
        None
    """
    ids = env_ids.cpu().tolist()

    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]

    data = robot.data
    # clone default states for selected envs
    default_q = data.default_joint_pos[env_ids].clone()
    default_vel = data.default_joint_vel[env_ids].clone()

    # find gripper joint index
    joint_names = data.joint_names
    gripper_idx = joint_names.index("Gripper")

    # set gripper to closed value
    default_q[:, gripper_idx] = gripper_closed_value

    # Optionally adjust other joints so EE is at height start_ee_height.
    # This may require a simple heuristic (e.g., lift shoulder) or actual IK.
    # Here we just assume one joint “Shoulder_Pitch” lifts and approximate:
    try:
        sp_idx = joint_names.index("Shoulder_Pitch")
        default_q[:, sp_idx] = default_q[:, sp_idx] - 0.5  # example adjustment
    except ValueError:
        pass

    # Write robot state for these environments
    robot.write_joint_state_to_sim(default_q, default_vel, env_ids=env_ids)
    robot.reset(env_ids=env_ids)

    # Now set object pose relative to the fixed gripper
    for idx in ids:
        # compute world transform of fixed gripper for this env
        prim_path = f"/World/envs/env_{idx}/Robot/Fixed_Gripper"
        stage = Usd.Stage.Open(Usd.Stage.GetDefaultLayer().GetIdentifier())  # or get context stage
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print("[WARNING]: Invalid prim path:", prim_path)
            continue
        xform = UsdGeom.Xformable(prim)
        mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        base_pos = mat.ExtractTranslation()
        qw, qx, qy, qz = mat.ExtractRotationQuat().GetReal(), mat.ExtractRotationQuat().GetImaginary()
        rot = Gf.Quatd(qw, qx, qy, qz)

        # compute object world offset
        local_off = Gf.Vec3d(*object_offset_local)
        world_off = Gf.Transform(rot, base_pos).TransformDir(local_off)
        world_off = world_off + Gf.Vec3d(0.0, 0.0, start_ee_height)

        target_pos = base_pos + world_off

        # build pose tensor
        pose_device = obj._device
        pose7 = torch.tensor([target_pos[0], target_pos[1], target_pos[2], qw, qx, qy, qz],
                              dtype=torch.float32, device=pose_device).unsqueeze(0)
        zero_vel = torch.zeros((1,6), dtype=torch.float32, device=pose_device)

        env_ids_tensor = torch.tensor([idx], dtype=torch.int32, device=env.device)
        obj.write_root_pose_to_sim(pose7, env_ids=env_ids_tensor)
        obj.write_root_velocity_to_sim(zero_vel, env_ids=env_ids_tensor)
        obj.reset(env_ids=env_ids)
    return None