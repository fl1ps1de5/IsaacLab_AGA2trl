import torch
import torch.nn as nn
from skrl.models.torch import Model, DeterministicMixin


class SimpleMLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device="cuda:0", clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, self.num_actions),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class BiggerMLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device="cuda:0", clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, self.num_actions),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
