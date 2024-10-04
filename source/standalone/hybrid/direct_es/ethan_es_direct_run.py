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
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# add relevant parser arguments
parser.add_argument(
    "--test",
    action="store_true",
    default=False,
    help="Determines wheteher to test/evaluate model (rather than train) .",
)

parser.add_argument(
    "--checkpoint", type=str, default=None, help="Specify .pth file to use as a training checkpoint (for evaluation)"
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


args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal

from functools import partial

import copy

from skrl.utils.model_instantiators.torch import shared_model

from omni.isaac.lab.envs import (
    ManagerBasedRLEnvCfg,
)

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config

from ethan_es_direct_trainer import DirectESTrainer

# begin configs
# define traininer configs - will be moved outside of this file
CARTPOLE_ES_TRAINER_CONFIG = {
    "num_generations": 100,
    "max_episode_length": 500,
    "sigma": 0.05,
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

DETERMINISTIC_ES = True


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

    return (mean_actions if DETERMINISTIC_ES else actions), log_prob, outputs


def apply_reparameterisation_patch(model):
    if not hasattr(model, "_original_act"):
        model._original_act = model.act
        model.act = partial(reparameterised_act, model)


def remove_reparameterisation_patch(model):
    if hasattr(model, "_original_act"):
        model.act = model._original_act
        del model._original_act


# process config taken from skrl runner
def process_cfg(cfg: dict) -> dict:
    """Convert simple types to skrl classes/components

    :param cfg: A configuration dictionary

    :return: Updated dictionary
    """
    _direct_eval = [
        "learning_rate_scheduler",
        "shared_state_preprocessor",
        "state_preprocessor",
        "value_preprocessor",
    ]

    def reward_shaper_function(scale):
        def reward_shaper(rewards, *args, **kwargs):
            return rewards * scale

        return reward_shaper

    def update_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in _direct_eval:
                    if type(d[key]) is str:
                        d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
                elif key in ["rewards_shaper_scale"]:
                    d["rewards_shaper"] = reward_shaper_function(value)
        return d

    return update_dict(copy.deepcopy(cfg))


def initialise_lazy_linear(module, input_shape):
    for name, child in module.named_children():
        if isinstance(child, nn.LazyLinear):
            # forward a dummy input to initialize the lazy layer
            dummy_input = torch.zeros(1, *input_shape)
            child(dummy_input)
        elif isinstance(child, nn.Sequential):
            # if it's a sequential container, we need to update input_shape as we go
            for sub_module in child:
                if isinstance(sub_module, nn.LazyLinear):
                    dummy_input = torch.zeros(1, *input_shape)
                    output = sub_module(dummy_input)
                    input_shape = output.shape[1:]
                elif hasattr(sub_module, "in_features") and hasattr(sub_module, "out_features"):
                    input_shape = (sub_module.out_features,)
        else:
            # recursively initialise nested modules
            initialise_lazy_linear(child, input_shape)
    # print model to check
    print(module)


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # set the environment seed
    # note: certain randomization occur in the environment initialization so we set the seed here
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, is_finite_horizon=False)

    # wrap environemtn with skrl wrapper for use with shared_model
    skrl_env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # remove 'class' key from cfg["models"]["policy"] and cfg["models"]["value"]
    try:
        del agent_cfg["models"]["policy"]["class"]
    except KeyError:
        pass
    try:
        del agent_cfg["models"]["value"]["class"]
    except KeyError:
        pass

    # we do not want a lazy value layer in our model as this ruins some ES functionality
    # obtain correct policy class based on skrl configs
    policy = shared_model(
        observation_space=skrl_env.observation_space,
        action_space=skrl_env.action_space,
        device=skrl_env.device,
        structure=None,
        roles=["policy", "value"],
        parameters=[
            process_cfg(agent_cfg["models"]["policy"]),
            process_cfg(agent_cfg["models"]["value"]),
        ],
    )

    initialise_lazy_linear(policy, policy.observation_space.shape)

    # we will now monkey-patch the policy to implement the reparametrisation trick
    # this allows the forward call to the policy to be differentiable and thus vectorisable
    apply_reparameterisation_patch(policy)

    # determine custom trainer config
    task_name = args_cli.task.split("-")[1].upper()

    # obtain trainer config
    config_name = f"{task_name}_ES_TRAINER_CONFIG"

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

    # define trainer log
    log_dir = "logs/custom/" + task_name.lower()
    trainer_config["logdir"] = log_dir

    # pass into trainer
    trainer = DirectESTrainer(cfg=trainer_config, env=env, policy=policy)

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
