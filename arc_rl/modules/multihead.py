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


class MultiheadMLP(nn.Module):
    """
    Multihead Cost Value MLP for predicting cost values.
    """

    def __init__(self, input_dim: int, output_dim: int = 3, hidden_dims=[128, 128]):
        super().__init__()
        layers = []
        in_dim = input_dim
        self.output_dim = output_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, output_dim)]
        self.multihead = nn.Sequential(*layers)

        print(f"Multihead MLP: {self.multihead}")

    def forward(self, critic_obs, masks=None, hidden_states=None) -> torch.Tensor:
        return self.multihead(critic_obs)
