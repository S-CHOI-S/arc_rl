# ARC RL

> [!note]
> This project is based on [RSL-RL](https://github.com/leggedrobotics/rsl_rl), developed by Robotic Systems Lab, ETH Zurich and contributors.

**Maintainer**: Sol Choi <br/>
**Affiliation**: Advanced Robot Control Lab, Korea Institute of Science and Technology (KIST) <br/>
**Contact**: solchoi@yonsei.ac.kr


## Setup

The package can be installed by cloning this repository and installing it with:

```bash
git clone https://github.com/S-CHOI-S/arc_rl
cd arc_rl
pip install -e .
```

The package supports the following logging frameworks which can be configured through `logger`:

* Tensorboard: https://www.tensorflow.org/tensorboard/
* Weights & Biases: https://wandb.ai/site
* Neptune: https://docs.neptune.ai/

For a demo configuration of PPO, please check the [dummy_config.yaml](config/dummy_config.yaml) file.
