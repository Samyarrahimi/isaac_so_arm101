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
# --------------- Pretrain: dex_cube -----------------------------
# ----------------------------------------------------------------

gym.register(
    id="SO-ARM100-Move-Object-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MoveObjectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Move-Object-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MoveObjectEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPORunnerCfg",
    },
    disable_env_checker=True,
)

# ----------------------------------------------------------------
# --------------- Finetune: all objects --------------------------
# ----------------------------------------------------------------

gym.register(
    id="SO-ARM100-Move-Object-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MoveObjectFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPOFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Move-Object-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MoveObjectFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPOFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

# ----------------------------------------------------------------
# --------------- Per-object fine-tuning envs --------------------
# ----------------------------------------------------------------

gym.register(
    id="SO-ARM100-Move-Bolt-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MoveBoltFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPOBoltFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Move-Bolt-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MoveBoltFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPOBoltFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Move-Gear-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MoveGearFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPOGearFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Move-Gear-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MoveGearFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPOGearFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Move-Nut-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MoveNutFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPONutFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Move-Nut-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MoveNutFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPONutFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Move-Peg-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MovePegFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPOPegFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Move-Peg-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MovePegFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPOPegFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Move-Box-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MoveBoxFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPOBoxFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM100-Move-Box-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100MoveBoxFinetuneEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MoveObjectPPOBoxFinetuneRunnerCfg",
    },
    disable_env_checker=True,
)
