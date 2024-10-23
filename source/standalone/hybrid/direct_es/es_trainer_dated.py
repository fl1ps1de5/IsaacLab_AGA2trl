import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from torch.distributions import Normal, kl_divergence

from torch import vmap
from torch._functorch.functional_call import functional_call

from torch.utils.tensorboard.writer import SummaryWriter
from utils.traininglogger import TrainingLogger
from utils import utils
from utils.adam import Adam

import numpy as np
import time
import copy
import os
import time


class ESTrainerDated(object):
    """
    Class which contains main training loop and associated
    """

    def __init__(self, cfg, policy, env):

        self.cfg = cfg
        self.env = env
        self.policy = policy.to(self.env.device)

        self.device = env.device

        self.model_arch = self._get_model_arch()
        self.num_params = self._get_num_params()

        self.num_gens = self.cfg["num_generations"]
        self.max_timesteps = self.cfg.get("max_timesteps", None)

        # initisalise hyperparameters
        self.sigma = self.cfg["sigma"]
        self.alpha = self.cfg["alpha"]
        self.sigma_decay = self.cfg["sigma_decay"]
        self.alpha_decay = self.cfg["alpha_decay"]
        self.sigma_limit = self.cfg["sigma_limit"]
        self.alpha_limit = self.cfg["alpha_limit"]
        self.kl_threshold = self.cfg.get("kl_threshold", 0)

        self.weight_decay = self.cfg.get("weight_decay", 0.01)

        self.npop = env.num_envs

        self.antithetic = self.cfg["antithetic"]

        self.mu = torch.zeros(self.num_params, device=self.device)
        # self.mu = torch.randn(self.num_params, device=self.device) * 0.01
        # self.mu = parameters_to_vector(self.policy.parameters()).to(self.device)

        # adam optimiser
        self.optimiser = Adam(self, stepsize=self.alpha)

        # obtain checkpoint if provided in config
        self.checkpoint = self.cfg.get("checkpoint", None)

        # set hybrid to True if checkpoint is provided
        self.hybrid = self.cfg.get("hybrid", False)

        endstring = "_hybrid_torch" if self.hybrid else "_es_torch"

        self.log_dir = os.path.join(self.cfg["logdir"], time.strftime("%Y-%m-%d_%H-%M-%S") + endstring)

        # initiate writer + save functionality
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.write_interval = self.cfg.get("write_interval", 20)  # write information every x timesteps/gens

        self.save_interval = self.cfg.get("save_interval", 5)  # save model every x generations
        self.save_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger = TrainingLogger(log_dir=self.log_dir)

        self.logger.log_setup(
            num_envs=self.npop,
            num_gens=self.num_gens,
            max_timesteps=self.max_timesteps,
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

        self.tracking_data = {}
        self.current_timestep = 0
        self.current_generation = 0

        # track best values
        self.best_mu = None
        self.best_reward = float("-inf")

        # rewards shaper
        self.rewards_shaper = cfg["rewards_shaper"]

        # initalise preprocessor if required
        self.state_preprocessor = self.cfg.get("state_preprocessor", None)
        if self.state_preprocessor is not None:
            self.state_preprocessor = self.state_preprocessor(size=self.env.observation_space, device=self.device)
        else:
            self.state_preprocessor = utils.empty_preprocessor

        # hybrid init
        if self.checkpoint:
            # load saved params
            saved_model = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(saved_model["policy"])

            # create inital policy to generate actions from it, for use with KL divergence
            self.prior_policy = copy.deepcopy(self.policy)

            if self.state_preprocessor is not utils.empty_preprocessor:
                self.state_preprocessor.load_state_dict(saved_model["state_preprocessor"])

            # update mu to be the current params
            self.mu = self._get_w().to(self.device)

    def _get_w(self):
        return parameters_to_vector(self.policy.parameters())

    def _get_num_params(self):
        w = parameters_to_vector(self.policy.parameters())
        return w.shape[0]

    def _get_model_arch(self):
        model_arch = copy.deepcopy(self.policy)
        model_arch = model_arch.to("meta")
        return model_arch

    def _reshape_params(self):
        params_dict = {}
        start_index = 0

        for name, param in self.model_arch.named_parameters():
            param_size = param.numel()
            new_shape = (self.npop,) + param.size()
            end_index = start_index + param_size

            params_dict[name] = self.pop_w[:, start_index:end_index].reshape(new_shape).to(self.device)

            start_index = end_index

        return params_dict

    @torch.no_grad()
    def _obtain_parallel_actions(self, states, params, base_model, hybrid):

        def fmodel(params, inputs):
            if hybrid:
                # calls reparameterised version of the act method to get actions
                # return log_prob for KL divergence
                actions, log_prob, outputs = functional_call(base_model, params, (inputs, "policy"))
                return actions, log_prob, outputs
            else:
                actions = functional_call(base_model, params, (inputs, "policy"))[0]
                return actions

        inputs = {"states": states} if isinstance(states, torch.Tensor) else states

        # use vmap to obtain actions in parallel
        if hybrid:
            actions, log_prob, outputs = vmap(fmodel, in_dims=(0, 0), randomness="different")(params, inputs)
            return actions, log_prob, outputs
        else:
            actions = vmap(fmodel, in_dims=(0, 0), randomness="different")(params, inputs)
            return actions

    @torch.no_grad()
    def generate_population(self):
        if self.antithetic:
            half_npop = self.npop // 2
            noise_half = torch.randn(half_npop, self.num_params, device=self.device)
            noise = torch.cat([noise_half, -noise_half], dim=0)
            # create multiplier for later, to ensure correct sign for antithetic noise
            self.multiplier = torch.cat([torch.ones(half_npop), -torch.ones(half_npop)]).to(self.device)
        else:
            noise = torch.randn(self.npop, self.num_params, device=self.device)
            self.multiplier = None

        self.noise = noise * self.sigma
        self.pop_w = self.mu.unsqueeze(0) + self.noise

    @torch.no_grad()
    def evaluate_unperturbed(self):
        """
        Evaluates the unperturbed policy (self.mu) using the same 4096 environments.
        This function assumes that the perturbed population has already been evaluated
        and that the environments are reset for a new evaluation of the unperturbed policy.
        """
        # Reset the environments for the unperturbed evaluation
        states, _ = self.env.reset()  # Same 4096 environments

        # Expand the unperturbed parameters (self.mu) to match the environment size
        params = {
            name: param.unsqueeze(0).expand(self.npop, *param.shape).to(self.device)
            for name, param in self.policy.named_parameters()
        }

        all_dones = torch.zeros(self.npop, dtype=torch.bool, device=self.device)
        sum_rewards = torch.zeros(self.npop, device=self.device)

        generation_steps = 0

        while not all_dones.all():
            states = self.state_preprocessor(states)

            # Get actions using the unperturbed parameters
            if self.hybrid:
                actions, _, _ = self._obtain_parallel_actions(
                    states=states,
                    params=params,
                    base_model=self.model_arch,
                    hybrid=self.hybrid,
                )
            else:
                actions = self._obtain_parallel_actions(
                    states=states,
                    params=params,
                    base_model=self.model_arch,
                    hybrid=self.hybrid,
                )

            # Step the environment with the unperturbed actions
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)
            sum_rewards += rewards.squeeze()

            dones = torch.bitwise_or(terminated, truncated)
            all_dones = torch.bitwise_or(all_dones, dones.squeeze())

            generation_steps += 1
            states = next_states

        # Return the accumulated rewards for the unperturbed evaluation
        return {
            "sum_rewards": sum_rewards.mean().item(),  # Use mean reward across all environments
            "generation_steps": generation_steps,
        }

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluates the current population weights within the simulation, in parallel
        """
        all_dones = torch.zeros(self.npop, dtype=torch.bool, device=self.device)
        sum_rewards = torch.zeros(self.npop).to(self.device)
        pop_kl_divergences = torch.zeros(self.npop).to(self.device)

        generation_steps = 0

        states, _ = self.env.reset()

        # reshape population params in order to be vectorised
        params = self._reshape_params()

        while not all_dones.all():

            states = self.state_preprocessor(states)

            if self.hybrid:

                actions, log_prob, outputs = self._obtain_parallel_actions(
                    states=states,
                    params=params,
                    base_model=self.model_arch,
                    hybrid=self.hybrid,
                )

                prior_actions, _, prior_outputs = self.prior_policy.act({"states": states}, role="policy")

                # calculate KL divergence
                with torch.no_grad():
                    prior_log_std = self.prior_policy.state_dict()["log_std_parameter"]
                    current_log_std = self.policy.state_dict()["log_std_parameter"]
                    prior_actions_dist = Normal(prior_outputs["mean_actions"], prior_log_std.exp())
                    current_actions_dist = Normal(outputs["mean_actions"], current_log_std.exp())
                    kl = kl_divergence(current_actions_dist, prior_actions_dist)
                    pop_kl_divergences += kl.mean(dim=1)

            else:
                actions = self._obtain_parallel_actions(
                    states=states,
                    params=params,
                    base_model=self.model_arch,
                    hybrid=self.hybrid,
                )

            # step environment
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)
            sum_rewards += rewards.squeeze()

            # track environments
            dones = torch.bitwise_or(terminated, truncated)
            all_dones = torch.bitwise_or(all_dones, dones.squeeze())

            # iterate steps in generation
            generation_steps += 1
            # count timestep
            self.current_timestep += 1
            # obtain next states
            states = next_states

        # divide kl's by steps
        pop_kl_divergences = pop_kl_divergences / generation_steps

        return sum_rewards, pop_kl_divergences

    @torch.no_grad()
    def fitness_shaping(self, rewards):
        """
        Rank transform rewards -> reduces the chances
        of falling into local optima early in training.
        """
        lamb = torch.tensor(rewards.size(0), dtype=torch.float32, device=self.device)
        sorted_rewards, sorted_indices = torch.sort(rewards, descending=True)
        ranks = torch.argsort(sorted_indices)

        # compute log terms (add one to ranks to ensure starting at 1)
        log_term = torch.log2(lamb / 2 + 1) - torch.log2(ranks + 1)
        shaped_rewards = torch.max(log_term, torch.tensor(0.0, device=self.device))

        # normalise rewards
        denom = shaped_rewards.sum()
        shaped_rewards = shaped_rewards / denom + 1 / lamb

        # reorder rewards to match original order
        final_shaped_returns = torch.zeros_like(rewards)
        final_shaped_returns[sorted_indices] = shaped_rewards

        return final_shaped_returns

    @torch.no_grad()
    def weight_update_plain(self, rewards):
        # compute fitness
        # rewards = self.fitness_shaping(rewards)
        # OR
        # normalise rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        if self.weight_decay > 0:
            l2_decay = utils.compute_weight_decay(self.weight_decay, self.pop_w)
            rewards += l2_decay

        # compute mu change
        mu_change = (1.0 / (self.npop * self.sigma)) * torch.matmul(self.noise.T, rewards)

        gradient = -mu_change

        # step optimiser
        self.optimiser.stepsize = self.alpha
        update_ratio = self.optimiser.update(gradient)

        # decay hyper params
        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

        if self.alpha > self.alpha_limit:
            self.alpha *= self.alpha_decay
            # for param_group in self.optimiser.param_groups:
            #     param_group["lr"] = self.alpha

    @torch.no_grad()
    def weight_update_with_trust_region(self, rewards, pop_kl_divergences):
        kl_divergence = pop_kl_divergences.mean()

        # compute fitness
        # rewards = self.fitness_shaping(rewards)
        # OR
        # normalise rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # weight decay
        if self.weight_decay > 0:
            l2_decay = utils.compute_weight_decay(self.weight_decay, self.pop_w)
            rewards += l2_decay

        # 2. compute mu change
        mu_change = (1.0 / (self.npop * self.sigma + 1e-8)) * torch.matmul(self.noise.T, rewards)
        gradient = -mu_change

        # 3. implement trust region
        proposed_update = gradient
        if kl_divergence > self.kl_threshold:
            scaling_factor = torch.sqrt(self.kl_threshold / kl_divergence)
            proposed_update *= scaling_factor

        # 4. assign gradient to mu and step optimiser
        self.optimiser.stepsize = self.alpha
        update_ratio = self.optimiser.update(proposed_update)

        # 5. adaptive sigma based on kl divergence
        if kl_divergence <= self.kl_threshold:
            self.sigma = min(self.sigma * 1.1, self.cfg["sigma"])
        elif kl_divergence > self.kl_threshold:
            self.sigma = max(self.sigma * 0.9, self.sigma_limit)

        # 7. decay alpha
        if self.alpha > self.alpha_limit:
            self.alpha *= self.alpha_decay

        # 8. update comparitive policy to be best from population
        # best_mu = self.pop_w[rewards.argmax()].clone()
        # vector_to_parameters(self.mu, self.prior_policy.parameters())

    def update_tracking_data(self, rewards, gen):
        self.tracking_data.update(
            {
                "Reward / Total reward (mean)": torch.mean(rewards).item(),
                "Reward / Total reward (max)": torch.max(rewards).item(),
                "Reward / Total reward (min)": torch.min(rewards).item(),
                "Action / Mean action": torch.mean(self.pop_w).item(),
                "Action / Std action": torch.std(self.pop_w).item(),
                "Action / Max action": torch.max(self.pop_w).item(),
                "Action / Min action": torch.min(self.pop_w).item(),
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

    def write_to_tensorboard(self, step: float) -> None:
        for tag, value in self.tracking_data.items():
            self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def load_flat_params(self, flat_params):
        """Loads flat parameters into policy network"""
        vector_to_parameters(flat_params, self.policy.parameters())

    def save_model(self, identifier):
        """Saves model parameters and other to a checkpoint"""
        save_path = os.path.join(self.save_dir, f"agent_{identifier}.pt")

        if identifier == "best":
            self.load_flat_params(self.best_mu)
        else:
            self.load_flat_params(self.mu)

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

        for gen in range(self.num_gens):
            # training loop
            it_time = time.time()

            # generate and evaluate population -> return rewards and log probs
            self.generate_population()
            rewards, pop_kl_divergences = self.evaluate()

            # leave this here for debugging and logging, but NOT used in main code
            # unpertrubed_reward = self.evaluate_unperturbed()["sum_rewards"]

            # shape rewards
            shaped_rewards = self.rewards_shaper(rewards)

            # perform weight update
            # self.weight_update_with_trust_region(shaped_rewards, pop_kl_divergences)
            self.weight_update_plain(shaped_rewards)

            # update training data
            self.update_tracking_data(rewards=rewards, gen=gen)

            # obtain mean reward from tracked data
            mean_reward = self.tracking_data["Reward / Total reward (mean)"]

            # print generation, mean reward and runtime
            print(
                f"Generation {gen}: Reward: {mean_reward:.2f}, "
                f"Timesteps: {self.current_timestep}, Time: {time.time() - it_time:.2f}"
            )

            # write every generation
            self.write_to_tensorboard(self.current_timestep)

            # save every 20th
            if gen % self.save_interval == 0 or gen == self.num_gens - 1:
                self.save_model(gen)

            # if new best reward
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.best_mu = self.mu.clone()
                # save best model
                self.save_model("best")
                # update best reward in logger file
                self.logger.update_best_reward(self.best_reward)

            # update logger file
            self.logger.update_timesteps(self.current_timestep)
            self.logger.update_generations(gen + 1)

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
        start_time = time.time()

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
