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
class ArcRlConstraintCfg:
    """Configuration for the Random Network Distillation (RND) module.

    For more information, please check the work from :cite:`schwarke2023curiosity`.
    """

    constraint_func: callable = MISSING
