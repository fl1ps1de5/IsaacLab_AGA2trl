import os
import sys

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="ES agent for Isaac Lab environemnts")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# add relevant parser arguments
parser.add_argument(
    "--test",
    action="store_true",
    default=False,
    help="Determines wheteher to test/evaluate model (rather than train) .",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Specify .pth file to use as a training checkpoint (for evaluation)",
)

parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal

from functools import partial

from skrl.envs.wrappers.torch import wrap_env
from skrl.utils.model_instantiators.torch import shared_model

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.skrl import process_skrl_cfg

from ethan_es_direct_trainer import DirectESTrainer

# begin configs
# define traininer configs - will be moved outside of this file
CARTPOLE_ES_TRAINER_CONFIG = {
    "num_generations": 100,
    "max_episode_length": 500,
    "sigma": 0.5,
    "sigma_decay": 1,
    "sigma_limit": 0.01,
    "alpha": 0.05,
    "alpha_decay": 1,
    "alpha_limit": 0.001,
    "checkpoint": None,
    "antithetic": True,
}

ANT_ES_TRAINER_CONFIG = {
    "num_generations": 500,
    "max_episode_length": 1000,
    "sigma": 0.1,
    "sigma_decay": 0.999,
    "sigma_limit": 0.01,
    "alpha": 0.01,
    "alpha_decay": 0.999,
    "alpha_limit": 0.001,
    "checkpoint": None,
}
# end configs


def reparameterised_act(self, inputs, role):
    mean_actions, log_std, outputs = self.compute(inputs, role)

    # clamp log standard deviations
    if self._clip_log_std:
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        std = torch.exp(log_std)

    self._log_std = log_std
    self._num_samples = mean_actions.shape[0]

    # create a distribution for use with log_prob computation
    self._distribution = Normal(mean_actions, log_std.exp())

    # obtain actions by using reparametrisation trick with no inplace randomness
    epsilon = torch.randn_like(mean_actions)
    actions = mean_actions + epsilon * log_std.exp()

    # clip actions
    if self._clip_actions:
        actions = torch.clamp(actions, min=self._clip_actions_min, max=self._clip_actions_max)

    # log of the probability density function
    log_prob = self._distribution.log_prob(inputs.get("taken_actions", actions))
    if self._reduction is not None:
        log_prob = self._reduction(log_prob, dim=-1)
    if log_prob.dim() != actions.dim():
        log_prob = log_prob.unsqueeze(-1)

    outputs["mean_actions"] = mean_actions
    return actions, log_prob, outputs


def apply_reparameterisation_patch(model):
    if not hasattr(model, "_original_act"):
        model._original_act = model.act
        model.act = partial(reparameterised_act, model)


def remove_reparameterisation_patch(model):
    if hasattr(model, "_original_act"):
        model.act = model._original_act
        del model._original_act


def main():
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, is_finite_horizon=False)

    # wrap environemtn with skrl wrapper for use with shared_model
    skrl_env = wrap_env(env)

    # determine custom trainer config
    task_name = args_cli.task.split("-")[1].upper()

    # obtain trainer config
    config_name = f"{task_name}_ES_TRAINER_CONFIG"

    # obtain correct policy class based on skrl configs
    policy = shared_model(
        observation_space=skrl_env.observation_space,
        action_space=skrl_env.action_space,
        device=skrl_env.device,
        structure=None,
        roles=["policy", "value"],
        parameters=[
            process_skrl_cfg(experiment_cfg["models"]["policy"], ml_framework=args_cli.ml_framework),
            process_skrl_cfg(experiment_cfg["models"]["value"], ml_framework=args_cli.ml_framework),
        ],
    )

    # we will now monkey-patch the policy to implement the reparametrisation trick
    # this allows the forward call to the policy to be differentiable and thus vectorisable
    apply_reparameterisation_patch(policy)

    # load the trainer config based on task name
    try:
        trainer_config = getattr(sys.modules[__name__], config_name)
    except AttributeError:
        raise ValueError(f"No config found for task: {task_name}")

    # give checkpoint path if we need it
    if args_cli.checkpoint:
        checkpoint_path = os.path.abspath(args_cli.checkpoint)

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found")
            return

        trainer_config["checkpoint"] = checkpoint_path

    # pass into trainer
    trainer = DirectESTrainer(cfg=trainer_config, env=skrl_env, policy=policy)

    # check if we should test
    if args_cli.test:
        trainer.test()
    # else train
    else:
        trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
