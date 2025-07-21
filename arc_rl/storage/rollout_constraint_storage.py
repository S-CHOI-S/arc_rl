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

from .rollout_storage import RolloutStorage


class RolloutConstraintStorage(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.costs = None
            self.cost_values = None

        def clear(self):
            super().clear()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
        constraints_shape=None,
    ):
        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
            rnd_state_shape=None,
            device="cpu",
        )
        self.constraints_shape = constraints_shape

        if self.training_type == "rl" and constraints_shape is not None:
            self.costs = torch.zeros(self.num_transitions_per_env, self.num_envs, *constraints_shape, device=self.device)
            self.cost_values = torch.zeros_like(self.costs)
            self.cost_returns = torch.zeros_like(self.costs)
            self.cost_advantages = torch.zeros_like(self.costs)

    def add_transitions(self, transition):
        super().add_transitions(transition)

        if self.training_type == "rl" and self.constraints_shape is not None:
            self.costs[self.step - 1].copy_(transition.costs)
            self.cost_values[self.step - 1].copy_(transition.cost_values)

    def compute_cost_returns(self, last_values, gamma, lam):
        if self.constraints_shape is None:
            raise RuntimeError("Constraint shape is None; cannot compute cost returns.")

        cost_advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.cost_values[step + 1]

            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.costs[step] + next_is_not_terminal * gamma * next_values - self.cost_values[step]
            cost_advantage = delta + next_is_not_terminal * gamma * lam * cost_advantage
            self.cost_returns[step] = cost_advantage + self.cost_values[step]

        self.cost_advantages = self.cost_returns - self.cost_values

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        base_generator = super().mini_batch_generator(num_mini_batches, num_epochs)

        if self.constraints_shape is None:
            for batch in base_generator:
                yield (*batch, None, None, None, None)
        else:
            costs = self.costs.flatten(0, 1)
            cost_values = self.cost_values.flatten(0, 1)
            cost_returns = self.cost_returns.flatten(0, 1)
            cost_advantages = self.cost_advantages.flatten(0, 1)

            batch_size = self.num_envs * self.num_transitions_per_env
            mini_batch_size = batch_size // num_mini_batches
            indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

            for epoch in range(num_epochs):
                for i in range(num_mini_batches):
                    start = i * mini_batch_size
                    end = (i + 1) * mini_batch_size
                    batch_idx = indices[start:end]

                    cost_batch = costs[batch_idx]
                    cost_values_batch = cost_values[batch_idx]
                    cost_returns_batch = cost_returns[batch_idx]
                    cost_advantages_batch = cost_advantages[batch_idx]

                    base_batch = next(base_generator)
                    yield (*base_batch, cost_batch, cost_values_batch, cost_returns_batch, cost_advantages_batch)

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        base_generator = super().recurrent_mini_batch_generator(num_mini_batches, num_epochs)

        if self.constraints_shape is None:
            for batch in base_generator:
                yield (*batch, None, None, None, None)
        else:
            for batch in base_generator:
                start, stop = batch[0].shape[1], batch[0].shape[1] + batch[1].shape[1]  # assume envs split
                cost_batch = self.costs[:, start:stop]
                cost_values_batch = self.cost_values[:, start:stop]
                cost_returns_batch = self.cost_returns[:, start:stop]
                cost_advantages_batch = self.cost_advantages[:, start:stop]
                yield (*batch, cost_batch, cost_values_batch, cost_advantages_batch, cost_returns_batch)
