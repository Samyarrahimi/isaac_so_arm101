# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ----------------------------------------------------------------
# --------------- LycheeAI live asset ----------------------------
# ----------------------------------------------------------------

gym.register(
    id="SO-ARM100-Grasp-Object-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspObjectEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Grasp-Object-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspObjectEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Grasp-Object-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspObjectFinetuneEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPOFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Grasp-Object-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspObjectFinetuneEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPOFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

# ----------------------------------------------------------------
# --------------- Per-object fine-tuning envs --------------------
# ----------------------------------------------------------------

gym.register(
    id="SO-ARM100-Grasp-Bolt-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspBoltFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPOBoltFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Grasp-Bolt-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspBoltFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPOBoltFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Grasp-Gear-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspGearFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPOGearFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Grasp-Gear-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspGearFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPOGearFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Grasp-Nut-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspNutFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPONutFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Grasp-Nut-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspNutFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPONutFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Grasp-Peg-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspPegFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPOPegFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Grasp-Peg-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspPegFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPOPegFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Grasp-Box-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspBoxFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPOBoxFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Grasp-Box-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100GraspBoxFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GraspPPOBoxFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)
