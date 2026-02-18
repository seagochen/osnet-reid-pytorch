"""
Learning rate scheduler with plateau detection.
"""


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.

    Args:
        optimizer: Wrapped optimizer
        mode: 'min' or 'max'
        factor: Factor by which LR is reduced (default: 0.5)
        patience: Number of epochs with no improvement (default: 5)
        min_lr: Lower bound on the learning rate (default: 1e-7)
    """

    def __init__(self, optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        self.best = None
        self.num_bad_epochs = 0
        self.num_reductions = 0

    def step(self, metric):
        """Update based on metric value."""
        if self.best is None:
            self.best = metric
            return

        if self._is_better(metric):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            self.num_reductions += 1

    def _is_better(self, metric):
        if self.mode == 'min':
            return metric < self.best
        return metric > self.best

    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr

    def state_dict(self):
        return {
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'num_reductions': self.num_reductions,
        }

    def load_state_dict(self, state_dict):
        self.best = state_dict['best']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.num_reductions = state_dict['num_reductions']
