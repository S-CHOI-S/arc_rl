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


class AuxiliaryMLP(nn.Module):
    """
    Auxiliary MLP for predicting targets.
    """

    def __init__(self, input_dim: int, output_dim: int = 3, hidden_dims=[32, 32]):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, output_dim)]
        self.auxiliary = nn.Sequential(*layers)

        print(f"Auxiliary MLP: {self.auxiliary}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.auxiliary(obs)


class AuxiliaryGRU(nn.Module):
    def __init__(self, observation_dim, hidden_dim, output_dim, n_layers):
        super(AuxiliaryGRU, self).__init__()
        
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.gru = nn.GRU(
            input_size=observation_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

        print(f"Auxiliary GRU: {self.gru}")

    def forward(self, x, hidden):
        """
        x: (batch_size, seq_len, observation_dim)
        hidden: (n_layers, batch_size, hidden_dim)
        returns: (batch_size, output_dim)
        """
        out, hidden = self.gru(x, hidden)        # out: (B, T, H)
        last_out = out[:, -1, :]                 # last timestep (B, H)
        prediction = self.fc(last_out)           # (B, output_dim)
        return prediction, hidden

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
