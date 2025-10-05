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