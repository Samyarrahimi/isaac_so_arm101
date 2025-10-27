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
import omni.usd
import random


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
        # find index of “Shoulder_Rotation”
        joint_names = robot.data.joint_names
        j_idx = joint_names.index("Shoulder_Rotation")
        default_joint_pos[idx, j_idx] = angle

    # Write the state for all envs
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    # Reset those envs
    robot.reset(env_ids=env_ids)
    return torch.ones((len(ids),), dtype=torch.bool, device=env.device)


def grasp_object(
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        gripper_closed_value: float = 0.1,
    ) -> None:
    """
    Initialize the episode start so that the object is already grasped by the robot.
    - Robot’s gripper joint is set to `gripper_closed_value`.
    - Robot’s joints set to default (with optional heuristic to raise EE).
    - Object pose is placed relative to the robot’s fixed gripper with `object_offset_local`.
    - End-effector (gripper) is approximately at `start_ee_height` above the table.
    
    Args:
        env: ManagerBasedRLEnv instance.
        env_ids: tensor of environment indices to apply.
        robot_cfg: SceneEntityCfg identifying the robot asset in scene.
        object_cfg: SceneEntityCfg identifying the object asset.
        gripper_closed_value: float value representing the gripper closed-state joint value.
        object_offset_local: (x, y, z) offset in the gripper’s local frame for object placement.
        start_ee_height: float extra height (world Z) to offset the object/gripper start.
    Returns:
        None
    """
    #print("grasp_object called with env_ids:", env_ids)
    # Ensure env_ids is a tensor on the correct device
    env_ids = env_ids.to(env.device, dtype=torch.int32)

    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]

    data = robot.data
    # Extract joint pos/vel only for the selected envs
    q   = robot.data.joint_pos[env_ids].clone()
    vel = robot.data.joint_vel[env_ids].clone()

    joint_names = data.joint_names
    # Find index for gripper joint
    gripper_idx = joint_names.index("Gripper")

    # Set gripper to closed value for all selected envs
    q[:, gripper_idx] = gripper_closed_value

    # Write robot joint state for selected envs
    robot.write_joint_state_to_sim(q, vel, env_ids=env_ids)
    robot.reset(env_ids=env_ids)
    return None

def check_released(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_moving"),
    force_threshold: float = 40.0
) -> torch.BoolTensor:
    """
    Returns a boolean tensor of shape [num_envs] indicating for each environment
    whether the object has been grasped. Uses a ContactSensor and checks if 
    the magnitude of forces in force_matrix_w exceeds `force_threshold`.
    
    Parameters
    ----------
    env : ManagerBasedRLEnv
        The RL environment instance.
    env_ids : torch.Tensor
        tensor of environment indices (shape [num_envs], dtype=torch.int32/int64).
    contact_sensor_cfg : SceneEntityCfg
        Configuration specifying which contact sensor to use.
    force_threshold : float
        Threshold on contact-force magnitude to declare a "grasp".
    """
    sensor = env.scene[contact_sensor_cfg.name]
    # force_matrix_w has shape (N_sensors, B, M, 3)
    f_mat_full = sensor.data.force_matrix_w  # torch.Tensor
    
    # We assume one contact sensor per env (N_sensors == num_envs) OR
    # else some mapping. Here we assume f_mat_full[env_index]
    # For simplicity, assume N_sensors == num_envs and sensors aligned with envs.
    
    # Now we collapse B and M dims: e.g., sum or max over them
    # Option: sum over B and M
    f_sum = f_mat_full.sum(dim=(1, 2))  # shape (num_envs, 3)
    
    # Compute magnitude:
    f_mag = torch.linalg.norm(f_sum, dim=-1)  # shape (num_envs,)

    #print("Force magnitudes:", f_mag)  # Debug print
    
    # boolean mask
    grasped_mask = f_mag > force_threshold  # shape (num_envs,)
    return ~grasped_mask.to(env.device)
