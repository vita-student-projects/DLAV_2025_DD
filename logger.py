import matplotlib.pyplot as plt

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

    def plot_metrics(self):
        """
        Plots the logged metrics.
        """ 

        for metric in self.all_metrics[0].keys():
          values = [m[metric] for m in self.all_metrics]
          plt.plot(self.steps, values, label=metric)
          plt.yscale("log")
        
