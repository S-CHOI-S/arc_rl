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

def OnPolicyRunnerFactory(env, train_cfg, log_dir=None, device="cpu"):
    alg_name = train_cfg["algorithm"]["class_name"]

    if alg_name in ("PPO", "Distillation"):
        from .on_policy_runner import OnPolicyRunner
        return OnPolicyRunner(env, train_cfg, log_dir, device)

    elif alg_name == "APPO":
        from .on_policy_runner_appo import OnPolicyRunnerAPPO
        return OnPolicyRunnerAPPO(env, train_cfg, log_dir, device)
    
    elif alg_name == "MIPO":
        from .on_policy_runner_mipo import OnPolicyRunnerMIPO
        return OnPolicyRunnerMIPO(env, train_cfg, log_dir, device)

    else:
        raise ValueError(f"Training type not found for algorithm {alg_name}.")
