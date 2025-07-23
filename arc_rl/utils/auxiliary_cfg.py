# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class ArcRlAuxiliaryCfg:
    """Configuration for the auxiliary tasks in the training.
    Auxiliary tasks are additional tasks that are trained alongside the main task.

    They can be used to improve the performance of the main task by providing additional
    supervision or by encouraging the agent to learn useful features.

    For example, auxiliary tasks can be used to encourage the agent to learn a
    representation of the environment that is useful for the main task.
    """

    class_name: str = "AuxiliaryMLP"
    """The class name for the auxiliary configuration. Default is [AuxiliaryMLP]."""

    input_dim: int = 45
    """The input dimension for the auxiliary training. Default is 45."""

    output_dim: int = 3
    """The output dimension for the auxiliary training. Default is 3."""

    hidden_dim: int = 32
    """The hidden dimension for the auxiliary training. Default is 32."""

    learning_rate: float = 1.0e-4
    """The learning rate for the auxiliary training. Default is 1.0e-4."""

    linvel_fcn: callable = MISSING
    """Function to compute the linear velocity auxiliary task. Must be defined in the training script."""
