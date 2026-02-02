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
    contact_sensor_moving_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_moving"),
    contact_sensor_fixed_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor_fixed"),
    force_threshold: float = 1.0,
    combine: str = "and",
) -> torch.BoolTensor:
    """
    Returns a boolean tensor of shape [num_envs] indicating for each environment
    whether the object has been grasped. Checks contact force on both the moving
    jaw and fixed gripper sensors.

    Parameters
    ----------
    env : ManagerBasedRLEnv
        The RL environment instance.
    contact_sensor_moving_cfg : SceneEntityCfg
        Configuration for the moving jaw contact sensor.
    contact_sensor_fixed_cfg : SceneEntityCfg
        Configuration for the fixed gripper contact sensor.
    force_threshold : float
        Threshold on contact-force magnitude to declare a "grasp".
    combine : str
        "and" requires both sensors above threshold, "or" requires either.
    """
    sensor_moving = env.scene[contact_sensor_moving_cfg.name]
    sensor_fixed = env.scene[contact_sensor_fixed_cfg.name]

    f_moving = sensor_moving.data.force_matrix_w
    f_fixed = sensor_fixed.data.force_matrix_w

    if f_moving is None or f_fixed is None:
        raise RuntimeError(
            "force_matrix_w is None for at least one sensor. "
            "Make sure ContactSensorCfg.filter_prim_paths_expr is set."
        )

    # Replace NaNs (no contact pairs) with 0
    f_moving = torch.nan_to_num(f_moving, nan=0.0)
    f_fixed = torch.nan_to_num(f_fixed, nan=0.0)

    # Sum over bodies (B) and filtered bodies (M) -> (N, 3)
    f_sum_moving = f_moving.sum(dim=(1, 2))
    f_sum_fixed = f_fixed.sum(dim=(1, 2))

    # Magnitude per env -> (N,)
    f_mag_moving = torch.linalg.norm(f_sum_moving, dim=-1)
    f_mag_fixed = torch.linalg.norm(f_sum_fixed, dim=-1)

    if combine.lower() == "and":
        grasped_mask = (f_mag_moving > force_threshold) & (f_mag_fixed > force_threshold)
    elif combine.lower() == "or":
        grasped_mask = (f_mag_moving > force_threshold) | (f_mag_fixed > force_threshold)
    else:
        raise ValueError("combine must be 'and' or 'or'")

    return grasped_mask.to(env.device)

