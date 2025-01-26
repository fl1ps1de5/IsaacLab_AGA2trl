import torch
from torch import vmap
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from torch.distributions import Normal, kl_divergence
from torch._functorch.functional_call import functional_call
from torch.utils.tensorboard.writer import SummaryWriter

from utils.traininglogger import TrainingLogger
from utils import utils
from utils.adam import Adam
from utils.layerwise_init import LayerwiseInitializer
from utils.hyperparam_manager import HyperParamManager

from typing import Tuple

import numpy as np
import time
import copy
import os
import time
import collections

SEED = 42


class CompleteESTrainer(object):
    """
    Class which contains main training loop and associated
    """

    def __init__(self, cfg, env, policy):

        # basic initialisation
        self.cfg = cfg
        self.env = env
        self.device = env.device
        self.policy = self._init_policy(policy)
        self.model_arch = self._get_model_arch()
        self.num_params = self._get_num_params()
        # time dependent hyperparameters
        self.num_gens = self.cfg["num_generations"]
        self.max_timesteps = self.cfg.get("max_timesteps", None)
        # define training hyperparameters
        self.sigma = self.cfg.get("sigma", 0.1)
        self.sigma_decay = self.cfg.get("sigma_decay", 1)
        self.sigma_limit = self.cfg.get("sigma_limit", 0)
        self.alpha = self.cfg.get("alpha", 0.01)
        self.alpha_decay = self.cfg.get("alpha_decay", 1)
        self.alpha_limit = self.cfg.get("alpha_limit", 0)
        self.kl_threshold = self.cfg.get("kl_threshold", 0)
        self.weight_decay = self.cfg.get("weight_decay", 0)
        self.antithetic = self.cfg.get("antithetic", False)
        # env hyperparameters
        self.npop = env.num_envs
        # pre-allocate tracking tensors
        self.sum_rewards = torch.zeros(self.npop, device=self.device)
        self.all_dones = torch.zeros(self.npop, dtype=torch.bool, device=self.device)
        # obtain testing / hybrid checkpoint
        self.checkpoint = self.cfg.get("checkpoint", None)
        self.hybrid = self.cfg.get("hybrid", False)
        # tracking data
        self.tracking_data = {}
        self.current_timestep = 0
        self.current_generation = 1  # start with first generation
        # track best values
        self.best_mu = None
        self.best_reward = float("-inf")
        # rewards shaper
        self.rewards_shaper = cfg["rewards_shaper"]
        # initialise mu
        if self.checkpoint is None:  # initalise mu to pretrained otherwise
            init_method = 1
            if init_method == 0:
                self.mu = torch.zeros(self.num_params, device=self.device)
            elif init_method == 1:
                self.mu = LayerwiseInitializer.initialize_flat_params(self.policy)
            self._load_flat_params(self.mu)
        # hybrid init
        else:
            self._checkpoint_setup()

        # intiialise optimiser
        self.optimiser = Adam(self, stepsize=self.alpha)
        # setuping writer and logger
        self._recording_setup()
        self.param_manager = HyperParamManager(self)

    @torch.no_grad()
    def _init_policy(self, policy):
        """Requires taking a single extra step in the environment in order to initalise the lazy layers of the policy
        This must be done before we obtain the model architecture, or further load flat population params into the policy
        """
        # initalise preprocessor if required
        self.state_preprocessor = self.cfg.get("state_preprocessor", None)
        if self.state_preprocessor is not None:
            self.state_preprocessor = self.state_preprocessor(size=self.env.observation_space, device=self.device)
        else:
            self.state_preprocessor = utils.empty_preprocessor

        policy.to(self.device)

        # changed based on 1.4.0
        states, _ = self.env.reset()
        states = self.state_preprocessor(states)
        policy.init_state_dict("policy", inputs={"states": states})
        policy.init_state_dict("value", inputs={"states": states})

        return copy.deepcopy(policy)

    def _recording_setup(self) -> None:
        """Sets up logging and writing functionality for trainer"""
        endstring = "_klespr_torch"
        npop_shorthand = f"{str(self.npop)[0]}k"
        log_string = f"{npop_shorthand}{SEED}_alpha_{self.alpha}_sigma_{self.sigma}_kl_{self.kl_threshold}_decay_{self.sigma_decay}"

        self.log_dir = os.path.join(self.cfg["logdir"], log_string + endstring)
        # initiate writer + save functionality
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.save_interval = self.cfg.get("save_interval", 5)  # save model every x generations
        self.save_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger = TrainingLogger(log_dir=self.log_dir)

        self.logger.log_setup(
            num_envs=self.npop,
            env_seed=self.env.seed,
            num_gens=self.num_gens,
            max_timesteps=self.max_timesteps,
            kl_threshold=self.kl_threshold,
            sigma=self.sigma,
            sigma_decay=self.sigma_decay,
            sigma_limit=self.sigma_limit,
            learning_rate=self.alpha,
            learning_rate_limit=self.alpha_limit,
            learning_rate_decay=self.alpha_decay,
            checkpoint=self.checkpoint,
            policy=self.policy,
            optimiser=("Adam" if self.optimiser is not None else "None"),
        )

    def _checkpoint_setup(self) -> None:
        """Setup checkpoint (used for hybrid or testing)
        Inludes loading saved model and setting up prior policy
        """
        # load saved params
        saved_model = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(saved_model["policy"])

        # create inital policy to generate actions from it, for use with KL divergence
        self.prior_policy = copy.deepcopy(self.policy)

        if self.state_preprocessor is not utils.empty_preprocessor:
            try:
                self.state_preprocessor.load_state_dict(saved_model["state_preprocessor"])
            except:
                pass

        # update mu to be the current params
        self.mu = self._get_w().to(self.device)

    def _get_w(self) -> torch.Tensor:
        """Sets the policy parameters to a vector "mu" and returns it

        Note we ensure that only "trainable" parameters are included in the vector

        Returns:
            torch.tensor: the flat parameters of trainer policy
        """
        return parameters_to_vector([p for p in self.policy.parameters() if p.requires_grad])

    def _get_num_params(self) -> int:
        """Count number of parameters in policy network

        Returns:
            int: number of parameters in policy  network
        """
        return sum(p.numel() for p in self.policy.parameters() if p.requires_grad)

    def _get_model_arch(self):
        """Obtain copy of model, and move it to meta device.
        This allows us to access a more efficent version of the policy object, which is filled with parameters later on.
        """
        model_arch = copy.deepcopy(self.policy)
        model_arch = model_arch.to("meta")
        return model_arch

    def _reshape_params(self, pop_w: torch.Tensor) -> dict:
        """Reshapes the population parameters (generated by generate_population) into a dictionary
        with suitable shape to be vectorsied. More specifically the shape is appropraite for pytorch functional_call
        """
        params_dict = {}
        start_index = 0

        print("Original pop_w shape:", pop_w.shape)

        for name, param in self.model_arch.named_parameters():

            param_size = param.numel()
            new_shape = (self.npop,) + param.size()
            end_index = start_index + param_size

            reshaped = pop_w[:, start_index:end_index].reshape(new_shape)
            params_dict[name] = reshaped

            start_index = end_index

        # # investigation on 23/1 - added this due to meta / trainable parameters behaviours
        # if "log_std_parameter" in self.policy.state_dict():
        #     real_log_std = self.policy.state_dict()["log_std_parameter"]  # e.g. shape [action_dim]
        #     # expand so each env / population index sees the same log_std
        #     # shape => (self.npop, *real_log_std.shape)
        #     expanded_log_std = real_log_std.unsqueeze(0).expand(self.npop, *real_log_std.shape).to(self.device)
        #     params_dict["log_std_parameter"] = expanded_log_std

        return params_dict

    @torch.no_grad()
    def _obtain_parallel_actions(self, states, params, base_model, hybrid):
        """Obtains actions from a dictionary of different parameters.
        One candidate of parameter vectors in params, correlates to one environment.

        We pretty much make a forward call to the policy with given states, and iterate the policy method to
        use all the parameters in the population. This is done in parallel using vmap.

        Args:
            states (torch.Tensor): states tensor from environment
            params (torch.Tensor): reshaped parameters (see _reshape_params)
            base_model (type(self.policy)): model architecture of self.policy (see _get_model_arch)
            hybrid (boolean): boolean to determine whether to return log_prob

        Returns:
            actions (torch.Tensor): tensor with shape [num_envs, action_dim]. Calculated actions for each environment, using each
                                    set of parameters.

            if hybrid flag is true also:

            log_prob (torch.Tensor): tensor with shape [num_envs, 1]. Log probability of the actions taken.
            outputs (dict): dictionary containing the mean actions and extra output values
        """

        def fmodel(params, inputs):
            if hybrid:
                # require outputs for hybrid in order to do KL calculations
                actions, _, outputs = functional_call(base_model, params, (inputs, "policy"))
                return actions, _, outputs
            else:
                # note we only grab the actions if not hybrid ([0] index)
                actions = functional_call(base_model, params, (inputs, "policy"))[0]
                return actions

        inputs = {"states": states} if isinstance(states, torch.Tensor) else states

        # use vmap to obtain actions in parallel
        if hybrid:
            actions, _, outputs = vmap(fmodel, in_dims=(0, 0), randomness="different")(params, inputs)
            return actions, _, outputs
        else:
            actions = vmap(fmodel, in_dims=(0, 0), randomness="different")(params, inputs)
            return actions

    @torch.no_grad()
    def _generate_population(self) -> torch.Tensor:
        """Generate population using CPU memory as buffer for large tensors"""
        # Generate samples on CPU
        samples = torch.randn(self.npop // 2, self.num_params, device="cpu")

        # Keep symmSamples on CPU until needed
        self.symmSamples = torch.zeros(self.npop, self.num_params, device="cpu")
        self.symmSamples[: self.npop // 2] = samples
        self.symmSamples[self.npop // 2 :] = -samples
        del samples

        # Move to GPU only for the computation
        symmSamples_gpu = self.symmSamples.to(self.device)
        pop_w = self.mu.unsqueeze(0) + self.sigma * symmSamples_gpu

        if self.hybrid and self.kl_threshold > 0:
            params = self._reshape_params(pop_w)

            states, _ = self.env.reset()
            states = self.state_preprocessor(states)

            _, _, current_outputs = self._obtain_parallel_actions(states, params, self.model_arch, hybrid=self.hybrid)
            _, _, prior_outputs = self.prior_policy.act({"states": states}, role="policy")

            kl = self._calculate_kl_divergence(prior_outputs, current_outputs)

            scaling_factors = torch.ones(self.npop, device=self.device)
            mask = kl > self.kl_threshold
            scaling_factors[mask] = self.kl_threshold / kl[mask]

            # Store scaled samples on CPU
            self.scaled_symmSamples = (symmSamples_gpu * scaling_factors.unsqueeze(1)).cpu()
            scaled_perturbations = self.sigma * self.scaled_symmSamples.to(self.device)
            pop_w = self.mu.unsqueeze(0) + scaled_perturbations

            del symmSamples_gpu, scaling_factors, params, current_outputs, prior_outputs, kl
            torch.cuda.empty_cache()

        return pop_w

    @torch.no_grad()
    def _calculate_kl_divergence(self, prior_outputs, current_outputs):
        prior_log_std = self.prior_policy.state_dict()["log_std_parameter"]
        current_log_std = self.policy.state_dict()["log_std_parameter"]

        prior_actions_dist = Normal(prior_outputs["mean_actions"], prior_log_std.exp())
        current_actions_dist = Normal(current_outputs["mean_actions"], current_log_std.exp())

        kl = kl_divergence(current_actions_dist, prior_actions_dist).mean(dim=1)
        print(f"Avg KL Divergence: {kl.mean().item()}")
        return kl

    @torch.no_grad()
    def _evaluate_population(self):
        """Generates and evaluates a population of parameter vectors by perturbing mu"""
        # Step 1: Generate population
        pop_w = self._generate_population()

        # Step 2: Evaluate population
        total_rewards = torch.zeros(self.npop, device=self.device)

        for trial in range(self.cfg.get("ntrials", 1)):
            rewards = self._rollout_population_hybrid(pop_w)

            total_rewards += rewards

        avg_rewards = total_rewards / self.cfg.get("ntrials", 1)

        # Step 3: return the average rewards over trials
        return avg_rewards

    @torch.no_grad()
    def _rollout_population_hybrid(self, pop_w: torch.Tensor) -> torch.Tensor:
        """Rollout the population through the environment and obtain summed rewards and KL.
        We obtain KL data as a way to manage policy updates"""
        assert self.hybrid is True, "Hybrid flag must be set to True to use this method"

        self.all_dones.zero_()
        self.sum_rewards.zero_()

        states, _ = self.env.reset()
        params = self._reshape_params(pop_w)

        while not self.all_dones.all():
            states = self.state_preprocessor(states, train=True)
            actions, log_prob, outputs = self._obtain_parallel_actions(states, params, self.model_arch, hybrid=True)

            next_states, rewards, terminated, truncated, infos = self.env.step(actions)
            self.current_timestep += 1

            dones = torch.bitwise_or(terminated, truncated)
            self.sum_rewards += rewards.squeeze()
            self.all_dones = torch.bitwise_or(self.all_dones, dones.squeeze())
            states = next_states

        return self.sum_rewards

    @torch.no_grad()
    def _compute_update_hybrid(self, fitness):

        # rank the fitness scores in descending order (higher is better)
        sorted_fitness, indices = torch.sort(fitness, descending=False)

        # assign ranks (0 to npop - 1)
        ranks = torch.zeros_like(fitness)
        ranks[indices] = torch.arange(self.npop, device=self.device).type_as(fitness)

        # compute utilities based on ranks
        utilities = ranks
        utilities /= self.npop - 1  # normalise ranks to [0, 1]
        utilities -= 0.5  # Shift to [-0.5, 0.5]

        # ensure utilities have zero mean
        utilities -= utilities.mean()

        # Move symmSamples to GPU only for matmul
        symmSamples_gpu = self.symmSamples.to(self.device)

        # compute the parameter update using utilities instead of raw fitnes
        mu_change = (1.0 / (self.npop * self.sigma)) * torch.matmul(symmSamples_gpu.t(), utilities)
        del symmSamples_gpu

        # compute the parameter update using utilities instead of raw fitness
        globalg = -mu_change + 0.005 * self.mu

        self.optimiser.stepsize = self.alpha
        update_ratio = self.optimiser.update(globalg)

        # decay sigma too
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_limit)

        del mu_change, globalg
        torch.cuda.empty_cache()

    def _update_tracking_data(self, rewards, gen):

        self.tracking_data.update(
            {
                "Reward / Total reward (mean)": torch.mean(rewards).item(),
                "Reward / Total reward (max)": torch.max(rewards).item(),
                "Reward / Total reward (min)": torch.min(rewards).item(),
                "Parameters / Mean": torch.mean(self.mu).item(),
                "Parameters / Std": torch.std(self.mu).item(),
                "Parameters / Max": torch.max(self.mu).item(),
                "Parameters / Min": torch.min(self.mu).item(),
                "Parameters / Norm": torch.norm(self.mu).item(),
                "Hyperparameters / sigma": self.sigma,
                "Hyperparameters / alpha": self.alpha,
                "Steps / steps taken": self.current_timestep,
                "Generations / Total current_generation": gen,
            }
        )

    def _post_generation(self, step, mean_reward):
        # # implementation details
        transiton_gen = self.cfg.get("transition_gen", None)
        if transiton_gen and self.current_generation > transiton_gen:
            self.param_manager.grow_kl()

        # logging details
        self._write_to_tensorboard(step)
        # save model every so often
        if self.current_generation % self.save_interval == 0 or self.current_generation == self.num_gens - 1:
            self._save_model(self.current_generation)
        # if new best reward then act accordingly
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.best_mu = self.mu.clone()
            # save best model
            self._save_model("best")
            # update best reward in logger file
            self.logger.update_best_reward(self.best_reward)
        # update logger file
        self.logger.update_timesteps(self.current_timestep)
        self.logger.update_generations(self.current_generation + 1)

    def _write_to_tensorboard(self, step: float) -> None:

        for tag, value in self.tracking_data.items():
            if not tag.startswith("Window /"):
                self.writer.add_scalar(tag, value, step)

        self.writer.flush()

    def _load_flat_params(self, flat_params):
        """Loads flat parameters into policy network"""
        vector_to_parameters(flat_params, self.policy.parameters())

    def _save_model(self, identifier):
        """Saves model parameters and other to a checkpoint"""
        save_path = os.path.join(self.save_dir, f"agent_{identifier}.pt")

        if identifier == "best":
            self._load_flat_params(self.best_mu)
        else:
            self._load_flat_params(self.mu)

        save_dict = {
            "policy": self.policy.state_dict(),
            # "state_preprocessor": self.state_preprocessor.state_dict(),  # COMMENT OUT WHEN RUNNING ES
            "mu": self.mu if identifier != "best" else self.best_mu,
            "sigma": self.sigma,
            "alpha": self.alpha,
            "current_timestep": self.current_timestep,
            "generation": identifier if isinstance(identifier, int) else -1,
        }

        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")

    def train(self):
        """
        Main training loop and saving the trained model
        """
        print("========Start Training========")
        start_time = time.time()
        best_track = {}
        worst_track = {}
        for gen in range(self.num_gens):
            # training loop
            it_time = time.time()
            # evaluate population in environment
            fitness = self._evaluate_population()
            # perform update
            if self.hybrid:
                self._compute_update_hybrid(fitness)
            # update training data
            self._update_tracking_data(rewards=fitness, gen=gen)
            # obtain mean reward from tracked data
            mean_reward = fitness.mean().item()
            # print generation results
            print(
                f"Generation {self.current_generation}: Reward: {mean_reward:.2f}, "
                f"Timesteps: {self.current_timestep}, Time: {time.time() - it_time:.2f}, "
            )
            self._post_generation(step=self.current_timestep, mean_reward=mean_reward)
            # if we are terminating after a certain amount of timesteps, then do so
            if self.max_timesteps:
                if self.current_timestep >= self.max_timesteps:
                    break
            self.current_generation += 1

        training_time = time.time() - start_time
        print("==========Training finished==========")
        print(f"Training time: {training_time:.2f} seconds")

        self.logger.finalize()

        # close writer
        self.writer.close()

    def test(self):
        """
        Testing loop using specified checkpoint - simply evaluates model
        """
        print("========Start Testing========")

        assert self.checkpoint is not None

        all_rewards = []

        ep_idx = 1

        while True:
            # reset the environment and initialize tracking for this evaluation
            all_dones = torch.zeros(self.npop, dtype=torch.bool, device=self.device)
            sum_rewards = torch.zeros(self.npop).to(self.device)

            # reset the environment
            states, _ = self.env.reset()

            # reshape parameters for use with vmap / functional call (all idendical)
            params = {
                name: param.unsqueeze(0).expand(self.npop, *param.shape).to(self.device)
                for name, param in self.policy.named_parameters()
            }

            while not all_dones.all():  # Keep running until all environments are done
                # obtain actions using the current policy parameters
                actions, _, _ = self._obtain_parallel_actions(
                    states=self.state_preprocessor(states),
                    params=params,
                    base_model=self.model_arch,
                    hybrid=self.hybrid,
                )

                # step through the environment
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                # sum rewards for the current evaluation
                sum_rewards += rewards.squeeze()

                dones = torch.bitwise_or(terminated, truncated)
                all_dones = torch.bitwise_or(all_dones, dones.squeeze())

                self.current_timestep += 1

                # move to the next state
                states = next_states

            # store the cumulative reward for this evaluation
            all_rewards.append(sum_rewards.mean().item())
            print(f"Episode {ep_idx}: Mean Reward: {sum_rewards.mean().item():.2f}")

            ep_idx += 1
