import os
import sys
import argparse
import copy

import torch
import torch.nn as nn
from torch.distributions import Normal
from functools import partial

# SKRL / IsaacLab imports
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils.model_instantiators.torch import shared_model
from skrl.resources.preprocessors.torch.running_standard_scaler import RunningStandardScaler

# Hydra config + IsaacLab tasks
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config

# Local code
from es_trainer_complete import CompleteESTrainer
from utils.config_loader import load_config, ConfigLoadError

###############################################################################
# Parse custom CLI arguments
###############################################################################
parser = argparse.ArgumentParser(description="ES agent for Isaac Lab environments")

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

parser.add_argument(
    "--test",
    action="store_true",
    default=False,
    help="Determines whether to test/evaluate model (rather than train).",
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

args_cli, unknown_args = parser.parse_known_args()


###############################################################################
# Reparameterised act patch (if you need to avoid rsample)
###############################################################################
def reparameterised_act(self, inputs, role):
    if role == "policy":
        mean_actions, log_std, outputs = self.compute(inputs, role)

        # clamp log standard deviations
        if self._clip_log_std:
            log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        self._log_std = log_std
        self._num_samples = mean_actions.shape[0]

        # build the distribution
        std = log_std.exp()
        self._distribution = Normal(mean_actions, std)

        # for pure ES, sample actions deterministically
        actions = mean_actions

        # clip actions
        if self._clip_actions:
            actions = torch.clamp(actions, min=self._clip_actions_min, max=self._clip_actions_max)

        # compute log_prob
        log_prob = self._distribution.log_prob(inputs.get("taken_actions", actions))
        if self._reduction is not None:
            log_prob = self._reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["mean_actions"] = mean_actions
        return actions, log_prob, outputs

    elif role == "value":
        # we only include the value role to initialize the lazy layers
        value, outputs = self.compute(inputs, role)
        return value, None, outputs


def apply_reparameterisation_patch(model):
    if not hasattr(model, "_original_act"):
        model._original_act = model.act
        model.act = partial(reparameterised_act, model)


def remove_reparameterisation_patch(model):
    if hasattr(model, "_original_act"):
        model.act = model._original_act
        del model._original_act


###############################################################################
# Process config function (unchanged)
###############################################################################
def process_cfg(cfg: dict) -> dict:
    """Convert simple types to skrl classes/components."""
    _direct_eval = [
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
                    if isinstance(d[key], str):
                        d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
                elif key in ["rewards_shaper_scale"]:
                    d["rewards_shaper"] = reward_shaper_function(value)
        return d

    return update_dict(copy.deepcopy(cfg))


###############################################################################
# Main function with Hydra (using env_cfg but overriding with load_isaaclab_env)
###############################################################################
@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, cfg: dict):
    """
    Main function that loads an IsaacLab environment, builds a policy, and runs ES.
    """
    # -------------------------------------------------------------------------
    # A) Create environment using load_isaaclab_env
    # -------------------------------------------------------------------------
    # Here we pass the user-specified task, num_envs, etc.
    # You can also pass unknown_args if IsaacLab might parse them (like --headless).
    env = load_isaaclab_env(
        task_name=args_cli.task,
        num_envs=args_cli.num_envs,
        cli_args=unknown_args,  # pass leftover CLI to IsaacLab
        show_cfg=True,
    )

    # Optionally set the environment seed
    if args_cli.seed is not None:
        env.reset(seed=args_cli.seed)

    # -------------------------------------------------------------------------
    # B) Wrap the environment for SKRL usage
    # -------------------------------------------------------------------------
    skrl_env = wrap_env(env, wrapper="isaaclab")

    # -------------------------------------------------------------------------
    # C) Prepare the policy via shared_model
    # -------------------------------------------------------------------------
    # remove 'class' key from the policy/value config if present
    for k in ("policy", "value"):
        cfg["models"][k].pop("class", None)

    policy = shared_model(
        observation_space=skrl_env.observation_space,
        action_space=skrl_env.action_space,
        device=skrl_env.device,
        structure=["GaussianMixin", "DeterministicMixin"],
        roles=["policy", "value"],
        parameters=[
            {
                **process_cfg(cfg["models"]["policy"]),
                "fixed_log_std": True,  # fix log_std for ES
                "initial_log_std": -4.6,  # ln(0.01)
            },
            process_cfg(cfg["models"]["value"]),
        ],
        single_forward_pass=True,
    )

    # Patch the policy to avoid in-place randomness if you need vmap or a deterministic rollout
    apply_reparameterisation_patch(policy)

    # -------------------------------------------------------------------------
    # D) Build the ES Trainer
    # -------------------------------------------------------------------------
    # Derive config type from CLI
    if args_cli.task is None:
        task_name = "unknown"
    else:
        # e.g. "Isaac-Ant" => split("-")[1] => "Ant" => .lower() => "ant"
        task_name = args_cli.task.split("-")[1].lower() if "-" in args_cli.task else args_cli.task

    config_type = "hybrid" if args_cli.hybrid else "es"

    try:
        trainer_config = load_config(task_name, config_type)
    except ConfigLoadError as e:
        print(f"Error loading configuration: {e}")
        env.close()
        return

    # If checkpoint provided
    if args_cli.checkpoint:
        checkpoint_path = os.path.abspath(args_cli.checkpoint)
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file not found: {checkpoint_path}")
            env.close()
            return
        trainer_config["checkpoint"] = checkpoint_path

    # define trainer log
    log_dir = "logs/custom/" + task_name.lower()
    trainer_config["logdir"] = log_dir

    # set hybrid flag
    trainer_config["hybrid"] = bool(args_cli.hybrid)

    # Add reward shaper from config if present
    agent_cfg = cfg["agent"].copy()
    agent_cfg["rewards_shaper"] = None
    agent_cfg.update(process_cfg(agent_cfg))
    trainer_config["rewards_shaper"] = agent_cfg["rewards_shaper"]

    # Construct the ES trainer
    trainer = CompleteESTrainer(cfg=trainer_config, env=skrl_env, policy=policy)

    # debug info
    print("\n" * 2 + "========================\n")
    print("Training policy with the following structure: ")
    print(policy)
    print(f"Policy training to complete {task_name} task")
    print(trainer_config, "\n")

    # -------------------------------------------------------------------------
    # E) Train or Test
    # -------------------------------------------------------------------------
    if args_cli.test:
        trainer.test()  # your test logic
    else:
        trainer.train()

    # close the environment
    env.close()


###############################################################################
# F) Entry point
###############################################################################
if __name__ == "__main__":
    main()
    # load_isaaclab_env sets up an atexit hook to close Omniverse
    # so we don't manually launch or close simulation_app
