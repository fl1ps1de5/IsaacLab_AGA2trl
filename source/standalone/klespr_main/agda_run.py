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
    "--checkpoint",
    type=str,
    default=None,
    help="Specify .pth file to use as a training checkpoint (for evaluation)",
)

parser.add_argument(
    "--hybrid",
    action="store_true",
    default=False,
    help="Adjusts logging + other func. to align with hybrid trainer approach)",
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
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch.running_standard_scaler import RunningStandardScaler

from omni.isaac.lab.envs import (
    ManagerBasedRLEnvCfg,
)

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config

from es_trainer_complete import CompleteESTrainer
from utils.config_loader import load_config, ConfigLoadError


def monkeypatched_act(self, inputs, role):
    if role == "policy":
        mean_actions, log_std, outputs = self.compute(inputs, role)

        # clamp log standard deviations
        if self._clip_log_std:
            log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)

        self._log_std = log_std
        self._num_samples = mean_actions.shape[0]

        # distribution (used only for log_prob calculation)
        self._distribution = Normal(mean_actions, log_std.exp())

        # manually sample actions with fixed variance
        epsilon = torch.randn_like(mean_actions)
        fixed_std = 0.01
        exploration_tensor = torch.zeros_like(log_std) + fixed_std

        # "sample actions" from distribution with fixed variance
        actions = mean_actions + epsilon * exploration_tensor

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

    elif role == "value":
        # we only include the value role in our ES implementation as it is neccecary to intialize the Lazy layers of the policy
        actions, outputs = self.compute(inputs, role)
        return actions, None, outputs


def triangularDistribution():
    # to do: implement triangular distribution
    pass


def apply_monkey_patch(model):
    if not hasattr(model, "_original_act"):
        model._original_act = model.act
        model.act = partial(monkeypatched_act, model)


def remove_monkey_patch(model):
    if hasattr(model, "_original_act"):
        model.act = model._original_act
        del model._original_act


# process config - taken from skrl runner
# modified to only require elements needed for ES
def process_cfg(cfg: dict) -> dict:
    """Convert simple types to skrl classes/components

    :param cfg: A configuration dictionary

    :return: Updated dictionary
    """
    _direct_eval = [
        # "learning_rate_scheduler",
        "state_preprocessor",
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


# removed for 1.4.0 update
# def initialise_lazy_linear(module, input_shape):
#     for name, child in module.named_children():
#         if isinstance(child, nn.LazyLinear):
#             # forward a dummy input to initialize the lazy layer
#             dummy_input = torch.zeros(1, *input_shape)
#             child(dummy_input)
#         elif isinstance(child, nn.Sequential):
#             # if it's a sequential container, we need to update input_shape as we go
#             for sub_module in child:
#                 if isinstance(sub_module, nn.LazyLinear):
#                     dummy_input = torch.zeros(1, *input_shape)
#                     output = sub_module(dummy_input)
#                     input_shape = output.shape[1:]
#                 elif hasattr(sub_module, "in_features") and hasattr(sub_module, "out_features"):
#                     input_shape = (sub_module.out_features,)
#         else:
#             # recursively initialise nested modules
#             initialise_lazy_linear(child, input_shape)


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, cfg: dict):

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # max iterations for training
    if args_cli.max_iterations:
        cfg["trainer"]["timesteps"] = args_cli.max_iterations * cfg["agent"]["rollouts"]
    cfg["trainer"]["close_environment_at_exit"] = False

    # set the environment seed
    # note: certain randomization occur in the environment initialization so we set the seed here
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else cfg["seed"]

    # # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, is_finite_horizon=False)

    # # wrap environemtn with skrl wrapper for use with shared_model
    skrl_env = wrap_env(env, wrapper="isaaclab")

    # remove 'class' key from cfg["models"]["policy"] and cfg["models"]["value"]
    try:
        del cfg["models"]["policy"]["class"]
    except KeyError:
        pass
    try:
        del cfg["models"]["value"]["class"]
    except KeyError:
        pass

    # obtain correct policy class based on skrl configs - updated for 1.4.0
    policy = shared_model(
        observation_space=skrl_env.observation_space,
        action_space=skrl_env.action_space,
        device=skrl_env.device,
        # structure=None, # to use default structure from 1.4.0 update
        roles=["policy", "value"],
        parameters=[
            process_cfg(cfg["models"]["policy"]),
            process_cfg(cfg["models"]["value"]),
        ],
        single_forward_pass=False,
    )

    # removing inplace randomness to make vmap happy and adjust action sampling
    apply_monkey_patch(policy)

    # determine custom trainer config
    task_name = args_cli.task.split("-")[1].lower()
    config_type = "hybrid" if args_cli.hybrid else "es"

    # load the trainer config based on task name
    try:
        trainer_config = load_config(task_name, config_type)
    except ConfigLoadError as e:
        print(f"Error loading configuration: {e}")
        simulation_app.close()
        return

    # give checkpoint path if we need it
    if args_cli.checkpoint:
        checkpoint_path = os.path.abspath(args_cli.checkpoint)
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            simulation_app.close()
            return
        trainer_config["checkpoint"] = checkpoint_path

    # define trainer log
    log_dir = "logs/custom/" + task_name.lower()
    trainer_config["logdir"] = log_dir

    # add hybrid flag into trainer config based on cli
    trainer_config["hybrid"] = True if args_cli.hybrid else False

    # add reward shaper into trainer config (defined through preprocessing)
    agent_cfg = cfg["agent"].copy()
    agent_cfg["rewards_shaper"] = None  # to avoid dictionary changed size during preprocess
    agent_cfg.update(process_cfg(agent_cfg))
    trainer_config["rewards_shaper"] = agent_cfg["rewards_shaper"]

    # pass into trainer (note which trainer im using atm)
    trainer = CompleteESTrainer(cfg=trainer_config, env=skrl_env, policy=policy)

    # pre-experiment outputs
    print("\n" * 3, "========================" + "\n")
    print("Training policy with following structure: ")
    print(policy)
    print(f"Policy training to complete {task_name} task")
    print(f"{trainer_config}\n")

    # check if we should test
    if args_cli.test:
        trainer.test()  # not implemented yet
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
