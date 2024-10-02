import torch
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
        self.max_length = cfg["max_episode_length"]

        self.sigma = cfg["sigma"]
        self.alpha = cfg["alpha"]
        self.sigma_decay = cfg["sigma_decay"]
        self.alpha_decay = cfg["alpha_decay"]
        self.sigma_limit = cfg["sigma_limit"]
        self.alpha_limit = cfg["alpha_limit"]

        # set antithetic to true if provided in config
        self.antithetic = bool(cfg["antithetic"])

        self.clip_obs = 10.0

        # obtain checkpoint if provided in config
        self.checkpoint = cfg.get("checkpoint", None)

        # set hybrid to True if checkpoint is provided
        self.hybrid = bool(self.checkpoint)

        self.npop = env.num_envs

        self.mu = torch.zeros(self.num_params, device=self.device)

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

        # states = self.env.reset()[0]["policy"]
        states, infos = self.env.reset()

        params = self._reshape_params()

        while not all_dones.all():  # and (episode_lengths <= self.max_length).any()

            actions = self._obtain_parallel_actions(states=states, params=params, base_model=self.model_arch)

            next_states, rewards, terminated, truncated, infos = self.env.step(actions)

            dones = torch.bitwise_or(terminated, truncated)
            all_dones = torch.bitwise_or(all_dones, dones.squeeze())

            # track non done envs
            active_envs = ~all_dones

            # sum reward for non-done envs
            sum_rewards[active_envs] += rewards.squeeze()[active_envs]

            # iterate episode length for non done envs
            episode_lengths[active_envs] += 1

            # states = next_states["policy"]
            states = next_states

        self.env.reset()

        return sum_rewards

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

            rewards = self.evaluate()

            self.weight_update_with_fitness_shaping(rewards)

            print(f"{gen}: {torch.mean(rewards)}, time: {time.time() - it_time}")

        training_time = time.time() - start_time
        print("==========Training finished==========")
        print(f"Training time: {training_time:.2f} seconds")

        # if self.save_model:
        #     # save params
        #     self.save_params("current_param.pth")

    def load_from_checkpoint(self):
        return

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
