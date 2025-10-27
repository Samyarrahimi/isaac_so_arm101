# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
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
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def check_grasped(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_moving"),
    force_threshold: float = 1.0
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
    
    # boolean mask
    grasped_mask = f_mag > force_threshold  # shape (num_envs,)
    return grasped_mask.to(env.device)

