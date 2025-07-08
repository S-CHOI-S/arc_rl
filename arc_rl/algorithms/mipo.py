# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ******************************************************************************
#  ARC_RL
#
#  Advanced Robot Control Reinforcement Learning Library
#
#      https://github.com/S-CHOI-S/arc_rl.git
#
#  Advanced Robot Control Lab. (ARC)
#  	  @ Korea Institute of Science and Technology
#
# 	  https://sites.google.com/view/kist-arc
#
# ******************************************************************************

# Authors: Sol Choi (Jennifer) #

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from arc_rl.modules import ActorCritic, MultiheadMLP
from arc_rl.modules.rnd import RandomNetworkDistillation
from arc_rl.storage import RolloutStorage
from arc_rl.utils import string_to_callable


class MIPO:
    """IPO: Interior-point Policy Optimization under Constraints (https://arxiv.org/abs/1910.09615)"""
    """Not Only Rewards But Also Constraints: Applications on Legged Robot Locomotion (https://arxiv.org/abs/2308.12517)."""

    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        # constraint_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        # Auxiliary parameters
        auxiliary_cfg: nn.Module | None = None,
        auxiliary_lr: float = 1e-3,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Constraint critic parameters
        num_constraints=0,
        initial_constraint_limits=None,
        constraint_value_loss_coef=1.0,
        temperature=20.0,
        constraint_limit_alpha=0.1,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # MIPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.actor_optimizer = optim.Adam(
            list(self.policy.actor.parameters()) + [self.policy.std],
            lr=actor_learning_rate,
        )  # actor optimizer
        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=critic_learning_rate)  # critic optimizer
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()
        self.num_constraints = num_constraints
        self.initial_constraint_limits = (
            torch.tensor(initial_constraint_limits, dtype=torch.float32, device=device)
            if initial_constraint_limits is not None
            else torch.zeros(num_constraints, dtype=torch.float32, device=device)
        )
        self.adaptive_constraint_limit = self.initial_constraint_limits
        # Create constraint critic
        self.constraint_critic = MultiheadMLP(
            input_dim=self.policy.critic_obs_dim,
            output_dim=self.num_constraints,
            # hidden_dims=self.policy.constraint_critic_hidden_dims,
        )
        self.constraint_critic.to(self.device)
        self.constraint_critic_optimizer = optim.Adam(
            self.constraint_critic.parameters(),
            lr=critic_learning_rate,
        ) # constraint critic optimizer

        # MIPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.constraint_value_loss_coef = constraint_value_loss_coef
        self.temperature = temperature
        self.constraint_limit_alpha = constraint_limit_alpha

        # Auxiliary parameters
        if auxiliary_cfg is not None:
            # Create auxiliary network
            self.auxiliary = auxiliary_cfg
            self.auxiliary.to(self.device)
            self.auxiliary_optimizer = torch.optim.Adam(self.auxiliary.parameters(), lr=auxiliary_lr)
        else:
            self.auxiliary = None
            self.auxiliary_optimizer = None

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        actions_shape,
        constraints_shape=None,
    ):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None

        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
            constraints_shape,
        )

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs

        # compute the cost values
        if self.constraint_critic is not None:
            self.transition.cost_values = self.constraint_critic(critic_obs).detach()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, costs=None):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        if costs is not None:
            # costs: Tensor of shape (num_envs, cost_dim)
            self.transition.costs = costs.clone()
        else:
            # if costs are not provided, we assume no costs
            self.transition.costs = torch.zeros(rewards.shape[0], 1, device=self.device)

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def compute_cost_returns(self, last_critic_obs):
        """Compute the cost returns for the constraints."""
        # compute cost returns
        last_cost_values = self.constraint_critic(last_critic_obs).detach()
        self.storage.compute_cost_returns(
            last_cost_values,
            self.gamma,
            self.lam,
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None
        # -- Constraint critic loss
        if self.constraint_critic is not None:
            mean_constraint_value_loss = 0
            with torch.no_grad():
                all_critic_obs = self.storage.privileged_observations  # shape: (T, num_envs, obs_dim)
                all_cost_values = self.constraint_critic(all_critic_obs)  # shape: (T, num_envs, num_constraints)

                # reshape to flat batch: (T * num_envs, num_constraints)
                all_cost_values = all_cost_values.view(-1, all_cost_values.shape[-1])
                mean_cost_values = all_cost_values.mean(dim=0)  # shape: (num_constraints,)

                initial = self.initial_constraint_limits.to(self.device).view(-1)
                self.adaptive_constraint_limit = torch.maximum(
                    initial,
                    mean_cost_values + self.constraint_limit_alpha * initial
                )
                print("Adaptive constraint limits:", self.adaptive_constraint_limit, self.adaptive_constraint_limit.shape)
        else:
            mean_constraint_value_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
            # for constraints
            target_cost_values_batch,
            cost_returns_batch,
            cost_advantages_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                    obs_type="policy",
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch,
                    actions=None,
                    env=self.symmetry["_env"],
                    obs_type="critic",
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
                if self.constraint_critic is not None:
                    # -- constraints
                    target_cost_values_batch = target_cost_values_batch.repeat(num_aug, 1, 1)
                    cost_returns_batch = cost_returns_batch.repeat(num_aug, 1, 1)
                    cost_advantages_batch = cost_advantages_batch.repeat(num_aug, 1, 1)


            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]
            # -- constraints
            if self.constraint_critic is not None:
                # compute the cost values
                cost_values_batch = self.constraint_critic(critic_obs_batch, masks=masks_batch)

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.actor_learning_rate = max(1e-5, self.actor_learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.actor_learning_rate = min(1e-2, self.actor_learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.actor_learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.actor_learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.actor_optimizer.param_groups:
                        param_group["lr"] = self.actor_learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Constraint critic loss
            if self.constraint_critic is not None:
                constraint_value_loss = (cost_values_batch - cost_returns_batch).pow(2).mean() # type: ignore

                # cost_values_batch: shape (batch_size, num_constraints)
                # adaptive_constraint_limit: shape (num_constraints,) or list of scalars
                diff = torch.clamp(self.adaptive_constraint_limit - cost_values_batch.detach(), min=1e-6) # type: ignore

                # Compute the log barrier
                mask = (cost_advantages_batch > 0).float() # type: ignore

                barrier = -torch.log(diff) / self.temperature  # shape: (batch_size, num_constraints)
                masked_barrier = barrier * mask

                constraint_barrier_loss = masked_barrier.mean()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch,
                        actions=None,
                        env=self.symmetry["_env"],
                        obs_type="policy",
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None,
                    actions=action_mean_orig,
                    env=self.symmetry["_env"],
                    obs_type="policy",
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:],
                    actions_mean_symm_batch.detach()[original_batch_size:],
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    actor_loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            if self.auxiliary is not None and self.auxiliary_optimizer is not None:
                base_velocity_target = critic_obs_batch[:, :3]

                auxiliary_pred = self.auxiliary(obs_batch.detach())
                auxiliary_loss = nn.functional.mse_loss(auxiliary_pred, base_velocity_target)

                self.auxiliary_optimizer.zero_grad()
                auxiliary_loss.backward()
                self.auxiliary_optimizer.step()

            # Compute the gradients
            # -- For MIPO
            # ----- Actor update -----
            self.actor_optimizer.zero_grad()
            actor_loss = surrogate_loss - self.entropy_coef * entropy_batch.mean()
            if self.constraint_critic is not None:
                actor_loss += constraint_barrier_loss
            actor_loss.backward()

            # ----- Critic update -----
            self.critic_optimizer.zero_grad()
            critic_loss = self.value_loss_coef * value_loss  # no entropy in critic
            critic_loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # -- For Constraints
            if self.constraint_critic is not None:
                self.constraint_critic_optimizer.zero_grad()
                constraint_value_loss = self.constraint_value_loss_coef * constraint_value_loss
                constraint_value_loss.backward() # type: ignore

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For MIPO
            # ----- Actor update -----
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            # ----- Critic update -----
            nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
            # -- For Constraints
            if self.constraint_critic_optimizer:
                nn.utils.clip_grad_norm_(self.constraint_critic.parameters(), self.max_grad_norm)
                self.constraint_critic_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # -- Constraint critic loss
            if mean_constraint_value_loss is not None:
                mean_constraint_value_loss += constraint_value_loss.detach().item()

        # -- For MIPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- For Constraints
        if self.constraint_critic is not None:
            mean_constraint_value_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.auxiliary:
            loss_dict["auxiliary"] = auxiliary_loss.item()
        if self.constraint_critic is not None:
            loss_dict["constraint_value_loss"] = mean_constraint_value_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
