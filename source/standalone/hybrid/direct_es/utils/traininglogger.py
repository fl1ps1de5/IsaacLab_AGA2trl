import time
import os


class TrainingLogger:
    def __init__(self, log_dir) -> None:
        self.log_dir = log_dir
        self.filename = os.path.join(self.log_dir, "notes.txt")

    def log_setup(self, **kwargs):
        with open(self.filename, "w") as f:
            f.write("Training Setup Details:\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            for key, value in kwargs.items():
                f.write(f"{key}: {value}\n")
            f.write("\n\nBest reward: (updated during training)")
            f.write("\nTimesteps taken: (updated during training)")
            f.write("\nCompleted generations: (updated during training)")

    def update_best_reward(self, reward):
        with open(self.filename, "r+") as f:
            content = f.read()
            f.seek(0)
            f.truncate()
            f.write(content.split("Best reward:")[0] + f"Best reward: {reward}\n")

    def update_timesteps(self, timesteps):
        with open(self.filename, "r+") as f:
            content = f.read()
            f.seek(0)
            f.truncate()
            f.write(content.split("Timesteps taken:")[0] + f"Timesteps taken: {timesteps}\n")

    def update_generations(self, timesteps):
        with open(self.filename, "r+") as f:
            content = f.read()
            f.seek(0)
            f.truncate()
            f.write(content.split("Completed generations:")[0] + f"Completed generations: {timesteps}\n")

    def write_update_method(self, method):
        with open(self.filename, "r+") as f:
            f.write(f"Update method: {method}\n")

    def finalize(self):
        with open(self.filename, "a") as f:
            f.write(f"\nEnd Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
