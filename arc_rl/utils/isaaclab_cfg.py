# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .rnd_cfg import ArcRlRndCfg
from .symmetry_cfg import ArcRlSymmetryCfg

#########################
# Policy configurations #
#########################


@configclass
class ArcRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    pass


@configclass
class ArcRlPpoActorCriticRecurrentCfg(ArcRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""

    pass


############################
# Algorithm configurations #
############################


@configclass
class ArcRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    pass


@configclass
class ArcRlAppoAlgorithmCfg:
    """Configuration for the APPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    actor_learning_rate: float = MISSING
    """The actor learning rate for the policy."""

    critic_learning_rate: float = MISSING
    """The critic learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. Default is False.

    If True, the advantage is normalized over the entire collected trajectories.
    Otherwise, the advantage is normalized over the mini-batches only.
    """

    symmetry_cfg: ArcRlSymmetryCfg | None = None
    """The symmetry configuration. Default is None, in which case symmetry is not used."""

    rnd_cfg: ArcRlRndCfg | None = None
    """The configuration for the Random Network Distillation (RND) module. Default is None,
    in which case RND is not used.
    """


#########################
# Runner configurations #
#########################


@configclass
class ArcRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    pass
