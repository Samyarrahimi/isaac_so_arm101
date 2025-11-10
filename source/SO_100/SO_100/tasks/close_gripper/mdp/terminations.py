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



def close_gripper_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """reset the gripper to close position."""
    robot = env.scene[robot_cfg.name]
    default_pos = robot.data.default_joint_pos[env_ids, :].clone()
    default_vel = robot.data.default_joint_vel[env_ids, :].clone()
    default_pos[:, -1] = 0.69 # gripper joint is the last one 
    print("default_pos after setting gripper to close:", default_pos)

    robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)
    robot.reset(env_ids=env_ids)
    return None

def randomize_gripper_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """reset the gripper to close position."""
    robot = env.scene[robot_cfg.name]
    default_pos = robot.data.default_joint_pos[env_ids, :].clone()
    default_vel = robot.data.default_joint_vel[env_ids, :].clone()
    default_pos[:, -1] = torch.rand(len(env_ids)) * (0.69 - 0.3) + 0.3 # gripper joint is the last one
    #print("default_pos after randomizing gripper:", default_pos)
    robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)
    robot.reset(env_ids=env_ids)
    return None

# def randomize_robot_joint_positions(
#         env: ManagerBasedRLEnv,
#         env_ids: torch.Tensor | list[int],
#         robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#         joint_noise_std: float = 0.05
#     ) -> torch.BoolTensor:
#     """
#     Randomize the robot's joint positions around their defaults *for the given env_ids*
#     within soft positional limits for each parallel environment.

#     Args:
#         env: ManagerBasedRLEnv instance.
#         env_ids: tensor or list of environment indices to apply the randomization to.
#         robot_cfg: SceneEntityCfg for the robot asset.
#         joint_noise_std: standard deviation of the noise to apply to the default joint positions.

#     Returns:
#         success_mask: torch.BoolTensor of shape (num_envs,) with True for envs that were
#                       randomized.
#     """
#     #print("randomize_robot_joint_positions called with env_ids:", env_ids)

#     robot = env.scene[robot_cfg.name]
#     data = robot.data

#     # Get the default joint positions/velocities for only those env_ids
#     default_q = data.default_joint_pos[env_ids].clone()    # shape = (len(env_ids), num_joints)
#     default_vel = data.default_joint_vel[env_ids].clone()  # same shape

#     # Sample noise, apply, and clamp to limits (also only for those env_ids)
#     noise = torch.randn_like(default_q) * joint_noise_std
#     perturbed_q = default_q + noise

#     limits = data.joint_pos_limits[env_ids]  # shape (len(env_ids), num_joints, 2)
#     lower = limits[..., 0]
#     upper = limits[..., 1]
#     perturbed_q = torch.max(torch.min(perturbed_q, upper), lower)

#     robot.write_joint_state_to_sim(perturbed_q, default_vel, env_ids=env_ids)
#     robot.reset(env_ids=env_ids)

#     success = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
#     return success