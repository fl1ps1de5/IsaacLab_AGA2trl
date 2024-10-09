import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch import vmap
from torch._functorch.functional_call import functional_call
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from utils.traininglogger import TrainingLogger
from utils.adam import Adam
from utils import utils

import numpy as np
import time
import copy
import sys
import os
import time

# at some point may need to make a base trainer class, but not for now


class DirectESTrainer(object):
    """
    Class which contains main training loop and associated
    """

    def __init__(self, cfg, policy, env):
        self.cfg = cfg
        self.env = env
        self.policy = policy

        self.device = env.device

        self.model_arch = self._get_model_arch()
        self.num_params = self._get_num_params()

        self.num_gens = self.cfg["num_generations"]
        self.max_episode_length = self.cfg["max_episode_length"]  # not implemented
        self.max_timesteps = self.cfg.get("max_timesteps", None)

        self.sigma = self.cfg["sigma"]
        self.alpha = self.cfg["alpha"]
        self.sigma_decay = self.cfg["sigma_decay"]
        self.alpha_decay = self.cfg["alpha_decay"]
        self.sigma_limit = self.cfg["sigma_limit"]
        self.alpha_limit = self.cfg["alpha_limit"]

        self.npop = env.num_envs

        self.mu = torch.zeros(self.num_params, device=self.device)

        # adam optimiser
        self.optimiser = Adam(self, self.alpha)

        # set antithetic to true if provided in config
        self.antithetic = self.cfg["antithetic"]

        # obtain checkpoint if provided in config
        self.checkpoint = self.cfg["checkpoint"]

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
            optimizer=("Adam" if self.optimiser is not None else "None"),
        )

        self.tracking_data = {}
        self.total_timesteps = 0

        # track best values
        self.best_mu = None
        self.best_reward = float("-inf")

        # initalise preprocessor if required
        self.state_preprocessor = self.cfg.get("state_preprocessor", None)
        if self.state_preprocessor:
            self.state_preprocessor = self.state_preprocessor(size=self.env.observation_space, device=self.device)
        else:
            self.state_preprocessor = utils.empty_preprocessor

        # initalise reward shaper (defined in config)
        self.reward_shaper = self.cfg["rewards_shaper"]

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
    def _obtain_parallel_actions(self, states, params, base_model):

        def fmodel(params, inputs):

            # calls reparameterised version of the act method to get actions
            actions, _, _ = functional_call(base_model, params, (inputs, "policy"))

            return actions

        inputs = {"states": states} if isinstance(states, torch.Tensor) else states

        # use vmap to obtain actions in parallel
        actions = vmap(fmodel, in_dims=(0, 0), randomness="different")(params, inputs)

        return actions

    def generate_population(self):
        if not self.antithetic:
            self.noise = torch.randn(self.npop, self.num_params).to(self.device)
            self.pop_w = self.mu.reshape(1, self.num_params) + self.noise * self.sigma
        else:
            half_npop = self.npop // 2
            self.noise_half = torch.randn(half_npop, self.num_params, device=self.device)
            self.noise = torch.cat([self.noise_half, -self.noise_half], dim=0)
            self.pop_w = self.mu.reshape(1, self.num_params) + self.noise * self.sigma

    def evaluate(self):
        """
        Evaluates the current population weights within the simulation, in parallel
        """
        all_dones = torch.zeros(self.npop, dtype=torch.bool, device=self.device)
        sum_rewards = torch.zeros(self.npop).to(self.device)
        episode_lengths = torch.zeros(self.npop, dtype=torch.long, device=self.device)

        states, _ = self.env.reset()

        # reshape population params in order to be vectorised
        params = self._reshape_params()

        while not all_dones.all():  # and (episode_lengths <= self.max_episode_length).any()

            actions = self._obtain_parallel_actions(
                states=self.state_preprocessor(states), params=params, base_model=self.model_arch
            )

            next_states, rewards, terminated, truncated, infos = self.env.step(actions)
            rewards = self.reward_shaper(rewards)
            self.total_timesteps += 1

            dones = torch.bitwise_or(terminated, truncated)
            all_dones = torch.bitwise_or(all_dones, dones.squeeze())

            # track non done envs
            active_envs = ~all_dones

            # sum reward for all envs
            sum_rewards += rewards.squeeze()

            # iterate episode length for non done envs
            episode_lengths[active_envs] += 1

            states = next_states

        return sum_rewards

    @torch.no_grad()
    def weight_update(self, rewards):
        """
        Conducts the weight updates on the raw parameters and decay sigma/alpha
        """
        self.logger.write_update_method("Normal")

        normal_reward = (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-8)

        mu_change = (1.0 / (self.npop * self.sigma)) * torch.matmul(self.noise.T, normal_reward)

        self.mu += self.alpha * mu_change

        if self.alpha > self.alpha_limit:
            self.alpha *= self.alpha_decay

        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

    @torch.no_grad()
    def weight_update_with_fitness_shaping(self, rewards):
        self.logger.write_update_method("Fitness shaping")

        # normalise rewards
        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-8)

        # rank rewards
        ranks = torch.argsort(torch.argsort(-rewards))

        # compute fitness scores
        fitness = (self.npop - ranks).float() / (self.npop - 1)

        # normalise fitness scores
        fitness = (fitness - fitness.mean()) / (fitness.std() + 1e-8)

        # compute mu update
        mu_change = (1.0 / (self.npop * self.sigma)) * torch.matmul(self.noise.T, fitness)

        # update mu
        self.mu += self.alpha * mu_change

        if self.alpha > self.alpha_limit:
            self.alpha *= self.alpha_decay

        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

    @torch.no_grad()
    def weight_update_with_adam(self, rewards):
        self.logger.write_update_method("Adam")

        normal_rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-8)
        mu_change = (1.0 / (self.npop * self.sigma)) * torch.matmul(self.noise.T, normal_rewards)

        self.optimiser.stepsize = self.alpha
        update_ratio = self.optimiser.update(-mu_change)

        # Decay sigma
        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

    def update_tracking_data(self, rewards, gen):
        self.tracking_data.update(
            {
                "Reward / Total reward (mean)": torch.mean(rewards).item(),
                "Reward / Total reward (max)": torch.max(rewards).item(),
                "Reward / Total reward (min)": torch.min(rewards).item(),
                "Hyperparameters / sigma": self.sigma,
                "Hyperparameters / alpha": self.alpha,
                "Steps / steps taken": self.total_timesteps,
                "Generations / Total generations": gen,
            }
        )

    def write_to_tensorboard(self, step: float) -> None:
        for tag, value in self.tracking_data.items():
            self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def load_flat_params(self, flat_params):
        vector_to_parameters(flat_params, self.policy.parameters())

    def save_model(self, identifier):
        save_path = os.path.join(self.save_dir, f"agent_{identifier}.pt")

        if identifier == "best":
            self.load_flat_params(self.best_mu)
        else:
            self.load_flat_params(self.mu)

        save_dict = {
            "policy": self.policy.state_dict(),
            "mu": self.mu if identifier != "best" else self.best_mu,
            "sigma": self.sigma,
            "alpha": self.alpha,
            "total_timesteps": self.total_timesteps,
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

        if self.hybrid:
            assert self.checkpoint is not None

            # load saved params
            saved_model = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(saved_model["policy"])
            self.state_preprocessor.load_state_dict(saved_model["state_preprocessor"])

            # update mu to be the current params
            self.mu = self._get_w().to(self.device)

        for gen in range(self.num_gens):
            # training loop
            it_time = time.time()

            # actual training loop
            self.generate_population()
            rewards = self.evaluate()
            self.weight_update_with_adam(rewards)

            # update training data
            self.update_tracking_data(rewards=rewards, gen=gen)

            # obtain mean reward from tracked data
            mean_reward = self.tracking_data["Reward / Total reward (mean)"]

            # print generation, mean reward and runtime
            print(
                f"Generation {gen}: Reward: {mean_reward:.2f}, "
                f"Timesteps: {self.total_timesteps}, Time: {time.time() - it_time:.2f}"
            )

            # write every generation (which is <=300 steps)
            self.write_to_tensorboard(self.total_timesteps)

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
            self.logger.update_timesteps(self.total_timesteps)
            self.logger.update_generations(gen + 1)

            # if we are terminating after a certain amount of timesteps, then do so
            if self.max_timesteps:
                if self.total_timesteps >= self.max_timesteps:
                    break

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

        if not self.hybrid:  # checkpoint was trained with ES
            saved_model = torch.load(self.checkpoint)
            self.policy.load_state_dict(saved_model["policy"])
            self.state_preprocessor.load_state_dict(saved_model["state_preprocessor"])
            # update mu to be the current params
            self.mu = self._get_w().to(self.device)

        else:  # checkpoint was trained with PPO
            # load saved params from PPO (basically evaluating PPO params in ES workflow)
            saved_model = torch.load(self.checkpoint)
            self.policy.load_state_dict(saved_model["policy"])
            self.state_preprocessor.load_state_dict(saved_model["state_preprocessor"])
            # update mu to be the current params
            self.mu = self._get_w().to(self.device)

        all_rewards = []

        ep_idx = 1

        while True:
            # Reset the environment and initialize tracking for this evaluation
            all_dones = torch.zeros(self.npop, dtype=torch.bool, device=self.device)
            sum_rewards = torch.zeros(self.npop).to(self.device)
            episode_lengths = torch.zeros(self.npop, dtype=torch.long, device=self.device)

            # Reset the environment
            states, _ = self.env.reset()

            # reshape parameters for use with vmap / functional call (all idendical)
            params = {
                name: param.unsqueeze(0).expand(self.npop, *param.shape).to(self.device)
                for name, param in self.policy.named_parameters()
            }

            while not all_dones.all():  # Keep running until all environments are done
                # Obtain actions using the current policy parameters
                actions = self._obtain_parallel_actions(
                    states=self.state_preprocessor(states), params=params, base_model=self.model_arch
                )

                # Step through the environment
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                rewards = self.reward_shaper(rewards)
                self.total_timesteps += 1

                dones = torch.bitwise_or(terminated, truncated)
                all_dones = torch.bitwise_or(all_dones, dones.squeeze())

                # Sum rewards for the current evaluation
                sum_rewards += rewards.squeeze()

                # Iterate episode length for non-done environments
                active_envs = ~all_dones
                episode_lengths[active_envs] += 1

                # Move to the next state
                states = next_states

            # Store the cumulative reward for this evaluation
            all_rewards.append(sum_rewards.mean().item())
            print(f"Episode {ep_idx}: Mean Reward: {sum_rewards.mean().item():.2f}")

            ep_idx += 1
