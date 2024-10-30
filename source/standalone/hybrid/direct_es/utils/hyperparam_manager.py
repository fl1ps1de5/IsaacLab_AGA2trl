import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector


class HyperParamManager:
    def __init__(self, trainer):
        self.trainer = trainer
        self.transition_gen = self.trainer.cfg.get("transition_gen", -1)
        self.base_kl_threshold = self.trainer.cfg.get("kl_threshold", 0)

        # Simple exploration parameters
        self.exploration_started = False
        self.sigma_growth_rate = 1.2
        self.kl_growth_rate = 2.0

        # Store initial values
        self.initial_sigma = self.trainer.sigma
        self.initial_kl = self.base_kl_threshold

    def transition_parameters(self):
        current_gen = self.trainer.current_generation

        if current_gen % self.transition_gen == 0:  # every 5 generations
            # update current mu into comparitive policy
            vector_to_parameters(self.trainer.mu, self.trainer.prior_policy.parameters())
            # reset sigma
            self.trainer.sigma = self.initial_sigma
            # reset kl threshold
            self.base_kl_threshold = self.initial_kl

    def begin_exploring(self):
        self.exploration_started = True
        while self.exploration_started:
            self.trainer.sigma *= self.sigma_growth_rate
            # self.trainer.kl_threshold = 0
            self.trainer.kl_threshold *= self.kl_growth_rate
