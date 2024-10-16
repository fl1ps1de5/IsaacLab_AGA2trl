class AdaptiveScheduler:
    def __init__(self, kl_threshold, min_lr=1e-5, max_lr=5e-3, lr_factor=1.2):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.kl_threshold = kl_threshold
        self.lr_factor = lr_factor

    def update(self, current_lr, kl_dist):
        # Adjust the learning rate based on the KL divergence
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / self.lr_factor, self.min_lr)
        elif kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * self.lr_factor, self.max_lr)

        return lr
