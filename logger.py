import matplotlib.pyplot as plt
import pandas as pd
import os

class Logger:
    def __init__(self):
        # Placeholder for potential future configs (e.g., log_dir, wandb_enabled, etc.)
        self.steps = []
        self.all_metrics = []

    def log(self, step=None, **metrics):
        """
        Logs the given metrics.

        Args:
            step (int, optional): The current step or epoch. Useful for tracking.
            **metrics: Arbitrary keyword arguments representing metric names and values.
        """
        prefix = f"[Step {step}] " if step is not None else ""
        metric_str = " | ".join(f"{k}: {v}" for k, v in metrics.items())

        self.steps.append(step)
        self.all_metrics.append(metrics)

        #print(prefix + metric_str)

    def plot_metrics(self, save_dir=None):
        """
        Plots the logged metrics.
        """ 

        # Create a subplot for each metrics
        fig, axs = plt.subplots(len(self.all_metrics[0].keys()), 1, figsize=(10, 10))
        axs = axs.flatten()
        for i, metric in enumerate(self.all_metrics[0].keys()):
            values = [m[metric] for m in self.all_metrics]
            axs[i].plot(self.steps, values)
            axs[i].set_xlabel("Step")
            axs[i].set_ylabel(metric)
            axs[i].grid(True)
            axs[i].set_title(metric)

        plt.tight_layout()
        plt.show()
        

        if save_dir:
            path = os.path.join(save_dir, "metrics.png")
            fig.savefig(path)

    def to_csv(self, dir):
        df = pd.DataFrame(self.all_metrics)
        df.to_csv(os.path.join(dir, "metrics.csv"))
        