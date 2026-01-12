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

    gripper_joint_ids, _ = robot.find_joints(["gripper"])
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
    gripper_joint_id = robot.find_joints(["gripper"])[0][0]
    grip_pos = robot.data.joint_pos[:, gripper_joint_id].squeeze(-1)  # (num_envs,)
    closed_enough = grip_pos < grip_threshold

    return contact_condition & closed_enough


def set_object_position(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
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
    #print("set_object_position called with env_ids:", env_ids)
    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]

    art_data = robot.data
    link_states = art_data.body_link_state_w  # shape (num_envs, num_links, 13)

    ids = env_ids.cpu().tolist()

    for idx in ids:
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
        pose7 = torch.tensor([target_pos[0], target_pos[1], target_pos[2],
                               qw, qx, qy, qz],
                              dtype=torch.float32,
                              device=env.device).unsqueeze(0)
        zero_vel = torch.zeros((1,6), dtype=torch.float32, device=env.device)

        env_ids_tensor = torch.tensor([idx], dtype=torch.int32, device=env.device)

        obj.write_root_pose_to_sim(pose7, env_ids=env_ids_tensor)
        obj.write_root_velocity_to_sim(zero_vel, env_ids=env_ids_tensor)
        obj.reset(env_ids=env_ids_tensor)
    return None


def randomize_robot_joint_positions(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | list[int],
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        joint_noise_std: float = 0.05
    ) -> torch.BoolTensor:
    """
    Randomize the robot's joint positions around their defaults *for the given env_ids*
    within soft positional limits for each parallel environment.

    Args:
        env: ManagerBasedRLEnv instance.
        env_ids: tensor or list of environment indices to apply the randomization to.
        robot_cfg: SceneEntityCfg for the robot asset.
        joint_noise_std: standard deviation of the noise to apply to the default joint positions.

    Returns:
        success_mask: torch.BoolTensor of shape (num_envs,) with True for envs that were
                      randomized.
    """
    #print("randomize_robot_joint_positions called with env_ids:", env_ids)

    robot = env.scene[robot_cfg.name]
    data = robot.data

    # Get the default joint positions/velocities for only those env_ids
    default_q = data.default_joint_pos[env_ids].clone()    # shape = (len(env_ids), num_joints)
    default_vel = data.default_joint_vel[env_ids].clone()  # same shape

    # Sample noise, apply, and clamp to limits (also only for those env_ids)
    noise = torch.randn_like(default_q) * joint_noise_std
    perturbed_q = default_q + noise

    limits = data.joint_pos_limits[env_ids]  # shape (len(env_ids), num_joints, 2)
    lower = limits[..., 0]
    upper = limits[..., 1]
    perturbed_q = torch.max(torch.min(perturbed_q, upper), lower)

    robot.write_joint_state_to_sim(perturbed_q, default_vel, env_ids=env_ids)
    robot.reset(env_ids=env_ids)

    success = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    return success


def randomize_shoulder_rotation(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
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
    #print("randomize_shoulder_rotation called with env_ids:", env_ids)

    robot = env.scene[robot_cfg.name]
    # Get the current default joint positions
    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel[env_ids].clone()

    ids = env_ids.cpu().tolist()
    num_ids = len(ids)

    # Loop through each env id and assign random value
    for idx in range(num_ids):
        angle = random.uniform(min_angle, max_angle)
        # find index of “shoulder_pan”
        joint_names = robot.data.joint_names
        j_idx = joint_names.index("shoulder_pan")
        default_joint_pos[idx, j_idx] = angle

    # Write the state for all envs
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    # Reset those envs
    robot.reset(env_ids=env_ids)
    return torch.ones((len(ids),), dtype=torch.bool, device=env.device)
