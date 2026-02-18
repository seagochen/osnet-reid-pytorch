"""
Exponential Moving Average for model weights.
"""
import copy
import math
import torch


class ModelEMA:
    """
    Maintains an exponential moving average of model parameters.

    Args:
        model: Model to track
        decay: EMA decay rate (default: 0.9999)
        tau: Ramp-up parameter (default: 2000)
    """

    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    def update(self, model):
        """Update EMA parameters."""
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))

        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
