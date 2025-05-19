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
    def __init__(self, obs_dim: int, output_dim: int = 2, hidden_dims=[32, 32]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
