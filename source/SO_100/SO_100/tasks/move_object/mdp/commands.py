import math
from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique
from isaaclab.utils.configclass import configclass


class UniformDiskPoseCommand(UniformPoseCommand):
    """Command term that samples (x, y) from a disk instead of a rectangle.

    Uses r = sqrt(uniform(r_min², r_max²)) for uniform area distribution —
    without the sqrt, samples cluster near the center.
    """

    cfg: "UniformDiskPoseCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        r_min, r_max = self.cfg.ranges.r_min, self.cfg.ranges.r_max

        # uniform area sampling over the annulus
        r = torch.empty(n, device=self.device).uniform_(r_min**2, r_max**2).sqrt()
        theta = torch.empty(n, device=self.device).uniform_(
            self.cfg.ranges.theta_min, self.cfg.ranges.theta_max
        )

        self.pose_command_b[env_ids, 0] = r * torch.cos(theta)  # x
        self.pose_command_b[env_ids, 1] = r * torch.sin(theta)  # y
        self.pose_command_b[env_ids, 2] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.ranges.pos_z
        )

        euler = torch.zeros(n, 3, device=self.device)
        euler[:, 0].uniform_(*self.cfg.ranges.roll)
        euler[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self.pose_command_b[env_ids, 3:] = (
            quat_unique(quat) if self.cfg.make_quat_unique else quat
        )


@configclass
class UniformDiskPoseCommandCfg(UniformPoseCommandCfg):
    """Samples target position uniformly from a disk (annulus) in the XY plane.

    Avoids wasting training episodes on unreachable rectangle corners when the
    robot's reachable workspace is roughly circular.
    """

    class_type: type = UniformDiskPoseCommand

    @configclass
    class Ranges:
        r_min: float = 0.15          # inner dead zone radius (m) — too close to reach
        r_max: float = 0.5           # max reach radius (m)
        theta_min: float = -math.pi  # full circle; restrict to a sector if needed
        theta_max: float = math.pi
        # kept to satisfy parent code paths (e.g. debug vis) — not used in sampling
        pos_x: tuple = (-0.5, 0.5)
        pos_y: tuple = (-0.5, 0.5)
        pos_z: tuple[float, float] = (0.05, 0.4)
        roll: tuple[float, float] = (0.0, 0.0)
        pitch: tuple[float, float] = (0.0, 0.0)
        yaw: tuple[float, float] = (0.0, 0.0)

    ranges: Ranges = Ranges()
