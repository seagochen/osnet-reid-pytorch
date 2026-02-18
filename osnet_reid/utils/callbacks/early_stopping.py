"""
Early stopping callback.
"""


class EarlyStopping:
    """
    Stop training when LR has been reduced too many times.

    Args:
        max_lr_reductions: Maximum number of LR reductions before stopping (default: 3)
        patience: Epochs without improvement before stop (default: 5)
    """

    def __init__(self, max_lr_reductions=3, patience=5):
        self.max_lr_reductions = max_lr_reductions
        self.patience = patience
        self.best = None
        self.counter = 0

    def should_stop(self, scheduler):
        """Check if training should stop based on scheduler state."""
        if scheduler.num_reductions >= self.max_lr_reductions:
            return True
        return False

    def step(self, metric, mode='min'):
        """Check if metric has improved."""
        if self.best is None:
            self.best = metric
            return False

        if mode == 'min':
            improved = metric < self.best
        else:
            improved = metric > self.best

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
