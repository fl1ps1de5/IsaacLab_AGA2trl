import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch import vmap
from torch.func import functional_call  # type: ignore
from torch.nn.utils import vector_to_parameters, parameters_to_vector
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

        self.num_gens = cfg["num_generations"]
        self.max_episode_length = cfg["max_episode_length"]

        self.sigma = cfg["sigma"]
        self.alpha = cfg["alpha"]
        self.sigma_decay = cfg["sigma_decay"]
        self.alpha_decay = cfg["alpha_decay"]
        self.sigma_limit = cfg["sigma_limit"]
        self.alpha_limit = cfg["alpha_limit"]

        self.npop = env.num_envs

        self.mu = torch.zeros(self.num_params, device=self.device)

        # set antithetic to true if provided in config
        self.antithetic = cfg["antithetic"]

        # obtain checkpoint if provided in config
        self.checkpoint = cfg["checkpoint"]

        # set hybrid to True if checkpoint is provided
        self.hybrid = cfg.get("hybrid", False)

        endstring = "_hybrid_torch" if self.hybrid else "_es_torch"

        self.log_dir = os.path.join(cfg["logdir"], time.strftime("%Y-%m-%d_%H-%M-%S") + endstring)

        # initiate writer + save functionality
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.write_interval = cfg.get("write_interval", 20)

        self.save_interval = cfg.get("save_interval", 20)
        self.save_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        self.tracking_data = {}
        self.total_timesteps = 0

        # track best values
        self.best_mu = None
        self.best_reward = float("-inf")

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
        steps_taken = 0

        states, _ = self.env.reset()
        # states = self.env.reset()[0]["policy"]

        params = self._reshape_params()

        while not all_dones.all():  # and (episode_lengths <= self.max_episode_length).any()

            actions = self._obtain_parallel_actions(states=states, params=params, base_model=self.model_arch)

            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            dones = torch.bitwise_or(terminated, truncated)
            all_dones = torch.bitwise_or(all_dones, dones.squeeze())

            # track non done envs
            active_envs = ~all_dones

            # sum reward for non-done envs
            sum_rewards[active_envs] += rewards.squeeze()[active_envs]

            # sum reward for all envs
            # sum_rewards += rewards

            # iterate episode length for non done envs
            episode_lengths[active_envs] += 1

            # iterate steps taken
            steps_taken += 1

            # states = next_states["policy"]
            states = next_states

        self.env.reset()

        return sum_rewards, steps_taken

    @torch.no_grad()
    def weight_update(self, rewards):
        """
        Conducts the weight updates on the raw parameters and decay sigma/alpha
        """
        normal_reward = (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-8)

        mu_change = (1.0 / (self.npop * self.sigma)) * torch.matmul(self.noise.T, normal_reward)

        self.mu += self.alpha * mu_change

        if self.alpha > self.alpha_limit:
            self.alpha *= self.alpha_decay

        if self.sigma > self.sigma_limit:
            self.sigma *= self.sigma_decay

    @torch.no_grad()
    def weight_update_with_fitness_shaping(self, rewards):

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

    def update_tracking_data(self, rewards, steps_taken, it_time):
        self.tracking_data.update(
            {
                "Reward / Total reward (mean)": torch.mean(rewards).item(),
                "Hyperparameters / sigma": self.sigma,
                "Hyperparameters / alpha": self.alpha,
                "Time / iteration": time.time() - it_time,
                "Timesteps / total": self.total_timesteps,
                "Timesteps / generation": steps_taken,
            }
        )

    def write_to_tensorboard(self, step: float) -> None:
        for tag, value in self.tracking_data.items():
            self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def load_flat_params(self, flat_params):
        vector_to_parameters(flat_params, self.policy.parameters())

    def save_model(self, identifier):
        save_path = os.path.join(self.save_dir, f"model_{identifier}.pt")

        if identifier == "best":
            self.load_flat_params(self.best_mu)
        else:
            self.load_flat_params(self.mu)

        save_dict = {
            "policy_state_dict": self.policy.state_dict(),
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
            saved_model = torch.load(self.checkpoint)["policy"]
            self.policy.state_dict().update({k: v for k, v in saved_model.items() if k in self.policy.state_dict()})

            # update mu to be the current params
            self.mu = self._get_w().to(self.device)

        for gen in range(self.num_gens):
            # training loop
            it_time = time.time()

            self.generate_population()
            rewards, steps_taken = self.evaluate()
            self.weight_update_with_fitness_shaping(rewards)

            # update total timesteps
            self.total_timesteps += steps_taken

            # log details into tensorboard
            self.update_tracking_data(rewards=rewards, steps_taken=steps_taken, it_time=it_time)

            # obtain mean reward from tracked data
            mean_reward = self.tracking_data["Reward / Total reward (mean)"]

            # print generation, mean reward and runtime
            print(
                f"Generation {gen}: Reward: {mean_reward:.2f}, "
                f"Timesteps: {self.total_timesteps}, Time: {self.tracking_data['Time / iteration']:.2f}"
            )

            if self.total_timesteps % self.write_interval == 0:
                self.write_to_tensorboard(self.total_timesteps)

            if gen % self.save_interval == 0 or gen == self.num_gens - 1:
                self.save_model(gen)

            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.best_mu = self.mu.clone()
                self.save_model("best")

        training_time = time.time() - start_time
        print("==========Training finished==========")
        print(f"Training time: {training_time:.2f} seconds")

        # save model parameters (into log dir pre_much)

        # close writer
        self.writer.close()

    def test(self):
        """
        Testing loop using specified checkpoint - simply evaluates model
        """
        print("broken rn")
        return
        # # Copy parameters from checkpoint policy to trainer policy
        # current_policy_dict = self.policy.state_dict()
        # saved_model = torch.load(checkpoint)["policy"]

        # for name, param in saved_model.items():
        #     if name in current_policy_dict:
        #         print(
        #             f"Shape for {name}: checkpoint {param.data.shape}, current {current_policy_dict[name].data.shape}"
        #         )
        #         current_policy_dict[name].data.copy_(param.data)

        # self.policy.load_state_dict(current_policy_dict)

        # self.w = parameters_to_vector(self.policy.parameters())
        # pop_w = self.w.repeat(self.npop, 1)
        # model_arch = self.model_arch

        # states = self.env.reset()[0]["policy"]

        # generation_rewards = []

        # params = reshape(self.env, pop_w, model_arch, self.device)

        # print("========Start Testing========")

        # for gen in range(self.num_gens):

        #     rewards = self.evaluate(pop_w)

        #     print(f"Reward: {torch.mean(rewards)}")

        # print("==========Testing finished==========")
        # # print(f"Testing time: {testing_time:.2f} seconds")
