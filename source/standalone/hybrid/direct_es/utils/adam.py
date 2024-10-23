# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py


import torch


class Optimizer(object):
    def __init__(self, pi):
        self.pi = pi
        self.dim = pi.num_params
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.pi.mu
        ratio = step.norm() / theta.norm()
        self.pi.mu = theta + step
        return ratio.item()

    def _compute_step(self, globalg):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999, epsilon=1e-08):
        super().__init__(pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = torch.zeros(self.dim, dtype=torch.float32, device=pi.mu.device)
        self.v = torch.zeros(self.dim, dtype=torch.float32, device=pi.mu.device)

    def _compute_step(self, globalg):
        t = torch.tensor(self.t, dtype=torch.float32, device=self.m.device)

        a = self.stepsize * torch.sqrt(1 - self.beta2**t) / (1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (torch.sqrt(self.v) + self.epsilon)
        return step
