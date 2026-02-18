"""
General utilities for OSNet ReID training.
"""
import random
import torch
import numpy as np
from pathlib import Path


def init_seeds(seed=0):
    """Initialize random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def colorstr(*args):
    """
    Colorize a string with ANSI escape codes.

    Usage:
        colorstr('bright_green', 'bold', 'Hello')
        colorstr('bright_red', 'Error!')
    """
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'bold': '\033[1m',
        'underline': '\033[4m',
        'end': '\033[0m',
    }

    *modifiers, string = args
    color_codes = ''.join(colors.get(m, '') for m in modifiers)
    return f'{color_codes}{string}{colors["end"]}'


def increment_path(path, exist_ok=False, sep=''):
    """
    Increment file or directory path (runs/exp -> runs/exp2, runs/exp3, ...).

    Args:
        path: Path to increment
        exist_ok: If True, return path as-is even if it exists
        sep: Separator between path and number

    Returns:
        Incremented path
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = [d for d in path.parent.glob(f"{path.stem}{sep}*") if d.is_dir()]
        if dirs:
            matches = [int(d.stem.replace(path.stem, '').replace(sep, '') or 0) for d in dirs]
            n = max(matches) + 1
        else:
            n = 2
        path = Path(f"{path}{sep}{n}{suffix}")
    path.mkdir(parents=True, exist_ok=True)
    return path
