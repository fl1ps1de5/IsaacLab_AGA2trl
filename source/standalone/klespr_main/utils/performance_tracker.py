import torch


class PerformanceTracker:
    def __init__(self, eval_interval: int = 200):
        """
        eval_interval: How many environment steps between performance evaluations
        """
        self.eval_interval = eval_interval
        self.last_eval_step = 0
        self.performance_history = []  # List of (step, performance) tuples

    def should_evaluate(self, current_step: int) -> bool:
        """Determine if we should do a performance evaluation"""
        return current_step - self.last_eval_step >= self.eval_interval

    def add_evaluation(self, step: int, rewards: torch.Tensor):
        """
        Add a performance evaluation point
        rewards: Tensor of rewards from evaluating the current policy
        """
        mean_performance = rewards.mean().item()
        self.performance_history.append((step, mean_performance))
        self.last_eval_step = step
