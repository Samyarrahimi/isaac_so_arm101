# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class MoveObjectPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 500
    experiment_name = "move_object"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MoveObjectPPOFinetuneRunnerCfg(MoveObjectPPORunnerCfg):
    """PPO config for fine-tuning from pretrained weights (all objects).

    Lower starting LR (3e-5) to reduce risk of overwriting pretrained features
    before the adaptive scheduler has a chance to react.
    3000 iterations for multi-object generalisation; save_interval=1000 gives
    3 checkpoints across the run.
    """
    experiment_name = "move_object_finetune"
    max_iterations = 3000
    save_interval = 1000
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ---------------------------------------------------------------------------
# Per-object fine-tuning runner configs.
# Each saves logs to its own experiment folder so runs stay separate.
# ---------------------------------------------------------------------------

@configclass
class MoveObjectPPOBoltFinetuneRunnerCfg(MoveObjectPPOFinetuneRunnerCfg):
    experiment_name = "move_bolt_finetune"
    max_iterations = 1500
    save_interval = 500


@configclass
class MoveObjectPPOGearFinetuneRunnerCfg(MoveObjectPPOFinetuneRunnerCfg):
    experiment_name = "move_gear_finetune"
    max_iterations = 1500
    save_interval = 500


@configclass
class MoveObjectPPONutFinetuneRunnerCfg(MoveObjectPPOFinetuneRunnerCfg):
    experiment_name = "move_nut_finetune"
    max_iterations = 1500
    save_interval = 500


@configclass
class MoveObjectPPOPegFinetuneRunnerCfg(MoveObjectPPOFinetuneRunnerCfg):
    experiment_name = "move_peg_finetune"
    max_iterations = 1500
    save_interval = 500


@configclass
class MoveObjectPPOBoxFinetuneRunnerCfg(MoveObjectPPOFinetuneRunnerCfg):
    experiment_name = "move_box_finetune"
    max_iterations = 1500
    save_interval = 500
