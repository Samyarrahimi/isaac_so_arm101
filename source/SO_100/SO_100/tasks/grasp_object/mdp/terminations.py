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
from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


from pxr import Gf
import random


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    diff_threshold: float = 0.01,
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    gripper_joint_ids, _ = robot.find_joints(["Gripper"])
    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(
            robot.data.joint_pos[:, gripper_joint_ids[0]]
            - torch.tensor(0.5, dtype=torch.float32).to(env.device)
        )
        > gripper_threshold,
    )
    return grasped

def object_is_grasped(
    env: ManagerBasedRLEnv,
    grip_threshold: float = 0.2,
    force_threshold: float = 0.1,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Check if object is grasped using ContactSensor.

    Conditions:
      - Contact force on gripper > threshold
      - Gripper joint is sufficiently closed
    """
    # --- contact force check ---
    sensor = env.scene[sensor_cfg.name]   # ContactSensor
    forces = sensor.data.net_forces_w     # (num_envs, B, 3)
    norms = torch.linalg.norm(forces, dim=-1)   # (num_envs, B)
    force_per_env = norms.max(dim=-1).values    # (num_envs,)
    contact_condition = force_per_env > force_threshold

    # --- gripper closure check ---
    robot = env.scene[robot_cfg.name]
    gripper_joint_id = robot.find_joints(["Gripper"])[0][0]
    grip_pos = robot.data.joint_pos[:, gripper_joint_id].squeeze(-1)  # (num_envs,)
    closed_enough = grip_pos < grip_threshold

    return contact_condition & closed_enough


def set_object_position(
        env,
        env_ids: torch.Tensor | list[int],
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        local_offset_xyz=(0.0, 0.0, 0.09),
        extra_z_lower=0.0
    ) -> None:
    """
    Set the object position in front of the robot's fixed gripper for the given env_ids.

    Args:
        env: ManagerBasedRLEnv instance.
        env_ids: tensor or list of environment indices to apply the reset to.
        robot_cfg: SceneEntityCfg for the robot asset.
        object_cfg: SceneEntityCfg for the object asset.
        local_offset_xyz: (x, y, z) offset in the robot fixed-gripper's local frame.
        extra_z_lower: extra downward offset (in world Z) to adjust height.
    Returns:
        None
    """
    # Ensure env_ids is a 1D list or tensor of ints
    if isinstance(env_ids, torch.Tensor):
        env_ids_list = env_ids.cpu().tolist()
    else:
        env_ids_list = list(env_ids)

    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]

    art_data = robot.data
    link_states = art_data.body_link_state_w  # shape (num_envs, num_links, 13)

    for idx in env_ids_list:
        # Fixed_Gripper is index -2
        pose_fixed = link_states[idx, -2]  

        pg = pose_fixed[:3]
        pg_x, pg_y, pg_z = float(pg[0]), float(pg[1]), float(pg[2])

        qg = pose_fixed[3:7]
        qw, qx, qy, qz = float(qg[0]), float(qg[1]), float(qg[2]), float(qg[3])

        base_pos = Gf.Vec3d(pg_x, pg_y, pg_z)
        rot      = Gf.Quatd(qw, qx, qy, qz)

        Tf = Gf.Transform()
        Tf.SetRotation(Gf.Rotation(rot))
        Tf.SetTranslation(base_pos)
        mat = Tf.GetMatrix()

        local_off = Gf.Vec3d(local_offset_xyz[0],
                             local_offset_xyz[1],
                             local_offset_xyz[2])
        world_off = mat.TransformDir(local_off)
        # apply extra offset in world Z
        world_off = world_off + Gf.Vec3d(0.0, 0.0, extra_z_lower)

        target_pos = base_pos + world_off

        # Build pose (x,y,z, qw, qx, qy, qz)
        pose_device = obj._device
        pose7 = torch.tensor([target_pos[0], target_pos[1], target_pos[2],
                               qw, qx, qy, qz],
                              dtype=torch.float32,
                              device=pose_device).unsqueeze(0)

        env_id_tensor = torch.tensor([idx], dtype=torch.int32, device=pose_device)
        zero_vel      = torch.zeros((1,6), dtype=torch.float32, device=pose_device)

        obj.write_root_pose_to_sim(pose7, env_ids=env_id_tensor)
        obj.write_root_velocity_to_sim(zero_vel, env_ids=env_id_tensor)
        obj.reset(env_ids=env_id_tensor)
    return None


def randomize_shoulder_rotation(
        env,
        env_ids: torch.Tensor | list[int],
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        min_angle: float = -1.56,
        max_angle: float =  1.56
    ) -> None:
    """
    Randomise the Shoulder_Rotation joint for each environment in env_ids with a
    different value in [min_angle, max_angle].

    Returns:
        None
    """
    robot = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    # Get the current default joint positions
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()

    # Make env_ids list
    if isinstance(env_ids, torch.Tensor):
        ids = env_ids.cpu().tolist()
    else:
        ids = list(env_ids)

    # Loop through each env id and assign random value
    for idx in ids:
        angle = random.uniform(min_angle, max_angle)
        # find index of “Shoulder_Rotation”
        joint_names = robot.data.joint_names
        j_idx = joint_names.index("Shoulder_Rotation")
        default_joint_pos[idx, j_idx] = angle

    # Write the state for all envs
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    # Reset those envs
    robot.reset(env_ids=torch.tensor(ids, dtype=torch.int32, device=env.device))

    # Return success flags
    return torch.ones((num_envs,), dtype=torch.bool, device=env.device)
