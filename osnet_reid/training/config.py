"""
Training configuration and OSNet model registry.
Centralized configuration management for OSNet ReID training.

Supports:
  - YAML config file (--config configs/reid.yaml)
  - CLI arguments override YAML values
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


# ============== Available Model Registry ==============
AVAILABLE_MODELS = {
    'osnet_x1_0': {'params': '2.2M', 'desc': 'OSNet x1.0 (standard)'},
    'osnet_x0_75': {'params': '1.3M', 'desc': 'OSNet x0.75'},
    'osnet_x0_5': {'params': '0.6M', 'desc': 'OSNet x0.5'},
    'osnet_x0_25': {'params': '0.2M', 'desc': 'OSNet x0.25 (ultra-light)'},
    'osnet_ibn_x1_0': {'params': '2.2M', 'desc': 'OSNet x1.0 + InstanceNorm (cross-domain)'},
}


def list_available_models() -> None:
    """Print all available OSNet models in a formatted table."""
    print("\n" + "=" * 70)
    print("Available OSNet Models for ReID")
    print("=" * 70)

    print(f"\n{'Model':<25} {'Params':<10} {'Description'}")
    print("-" * 65)
    for name, info in AVAILABLE_MODELS.items():
        print(f"  {name:<23} {info['params']:<10} {info['desc']}")

    print("\n" + "=" * 70)
    print("Usage: python scripts/train.py --config configs/reid.yaml --arch osnet_x1_0")
    print("=" * 70 + "\n")


def validate_model(name: str) -> bool:
    """Validate if a model name is valid."""
    return name in AVAILABLE_MODELS


def load_yaml(cfg_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _deep_update(base: Dict, override: Dict) -> Dict:
    """Recursively update base dict with override dict (override wins)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def build_config(args) -> Dict[str, Any]:
    """
    Build configuration dictionary.

    Priority: CLI arguments > YAML config > defaults.

    Args:
        args: Parsed command line arguments

    Returns:
        Configuration dictionary
    """
    # Start with YAML config if provided
    cfg_path = getattr(args, 'config', '') or getattr(args, 'cfg', '')
    if cfg_path:
        config = load_yaml(cfg_path)
        config.setdefault('model', {})
        config.setdefault('data', {})
        config.setdefault('train', {})
        config.setdefault('val', {})
        config.setdefault('output', {})
        config.setdefault('advanced', {})
    else:
        config = {
            'model': {},
            'data': {},
            'train': {},
            'val': {},
            'output': {},
            'advanced': {},
        }

    # CLI overrides
    cli_overrides = _collect_cli_overrides(args)
    _deep_update(config, cli_overrides)

    # Fill in defaults
    m = config['model']
    m.setdefault('arch', 'osnet_x1_0')
    m.setdefault('pretrained', True)
    m.setdefault('num_classes', 0)  # auto-detected from dataset
    m.setdefault('reid_dim', 512)

    d = config['data']
    d.setdefault('root', '')
    d.setdefault('csv', 'labels.csv')
    d.setdefault('num_instances', 4)
    d.setdefault('task', 'reid')  # 'reid' or 'face'

    # Task-aware defaults for input_size and augmentation
    task = d['task']
    if task == 'face':
        m.setdefault('input_size', [112, 112])
    else:
        m.setdefault('input_size', [256, 128])

    aug = d.setdefault('augmentation', {})
    if task == 'face':
        aug.setdefault('pad', 4)
        aug.setdefault('random_rotation', 10)
    else:
        aug.setdefault('pad', 10)
        aug.setdefault('random_rotation', 0)
    aug.setdefault('random_flip', 0.5)
    aug.setdefault('brightness', 0.2)
    aug.setdefault('contrast', 0.15)
    aug.setdefault('saturation', 0.1)
    aug.setdefault('hue', 0.0)
    aug.setdefault('random_erasing', 0.5)
    aug.setdefault('erasing_scale', [0.02, 0.4])

    t = config['train']
    t.setdefault('epochs', 60)
    t.setdefault('lr', 3.5e-4)
    t.setdefault('backbone_lr', 3.5e-5)
    t.setdefault('batch_size', 64)
    t.setdefault('num_workers', 4)
    t.setdefault('weight_decay', 5e-4)
    t.setdefault('label_smooth', 0.1)
    t.setdefault('warmup_epochs', 5)
    t.setdefault('loss_type', 'triplet')
    t.setdefault('metric_weight', 1.0)
    t.setdefault('triplet_margin', 0.3)
    t.setdefault('circle_margin', 0.25)
    t.setdefault('circle_scale', 64)

    v = config['val']
    v.setdefault('val_split', 0.2)

    o = config['output']
    o.setdefault('project', 'runs/train')
    o.setdefault('name', 'exp')

    a = config['advanced']
    a.setdefault('ema', False)
    a.setdefault('seed', 0)
    a.setdefault('device', '')

    return config


# Mapping: CLI arg name -> (config section, config key)
_CLI_MAP = {
    'arch':             ('model', 'arch'),
    'no_pretrained':    ('model', 'pretrained'),  # inverted
    'num_classes':      ('model', 'num_classes'),
    'reid_dim':         ('model', 'reid_dim'),
    'img_height':       ('model', 'input_size'),  # combined with img_width
    'task':             ('data', 'task'),
    'data_root':        ('data', 'root'),
    'csv':              ('data', 'csv'),
    'num_instances':    ('data', 'num_instances'),
    'epochs':           ('train', 'epochs'),
    'lr':               ('train', 'lr'),
    'backbone_lr':      ('train', 'backbone_lr'),
    'batch_size':       ('train', 'batch_size'),
    'workers':          ('train', 'num_workers'),
    'weight_decay':     ('train', 'weight_decay'),
    'triplet_margin':   ('train', 'triplet_margin'),
    'label_smooth':     ('train', 'label_smooth'),
    'metric_weight':    ('train', 'metric_weight'),
    'warmup_epochs':    ('train', 'warmup_epochs'),
    'loss_type':        ('train', 'loss_type'),
    'circle_margin':    ('train', 'circle_margin'),
    'circle_scale':     ('train', 'circle_scale'),
    'val_split':        ('val', 'val_split'),
    'project':          ('output', 'project'),
    'name':             ('output', 'name'),
    'ema':              ('advanced', 'ema'),
    'seed':             ('advanced', 'seed'),
    'device':           ('advanced', 'device'),
}

# argparse defaults
_ARGPARSE_DEFAULTS = {
    'arch': 'osnet_x1_0',
    'no_pretrained': False,
    'num_classes': 0,
    'reid_dim': 512,
    'img_height': 256,
    'img_width': 128,
    'task': 'reid',
    'data_root': '',
    'csv': 'labels.csv',
    'num_instances': 4,
    'epochs': 60,
    'lr': 3.5e-4,
    'backbone_lr': 3.5e-5,
    'batch_size': 64,
    'workers': 4,
    'weight_decay': 5e-4,
    'triplet_margin': 0.3,
    'label_smooth': 0.1,
    'metric_weight': 1.0,
    'warmup_epochs': 5,
    'loss_type': 'triplet',
    'circle_margin': 0.25,
    'circle_scale': 64,
    'val_split': 0.2,
    'project': 'runs/train',
    'name': 'exp',
    'ema': False,
    'seed': 0,
    'device': '',
}


def _collect_cli_overrides(args) -> Dict[str, Any]:
    """Collect CLI arguments that were explicitly passed (differ from defaults)."""
    overrides: Dict[str, Any] = {}

    for arg_name, (section, key) in _CLI_MAP.items():
        if not hasattr(args, arg_name):
            continue
        val = getattr(args, arg_name)
        default = _ARGPARSE_DEFAULTS.get(arg_name)

        if val == default:
            continue

        # Special handling
        if arg_name == 'no_pretrained':
            val = not val
        elif arg_name == 'img_height':
            img_w = getattr(args, 'img_width', 128)
            val = [val, img_w]

        overrides.setdefault(section, {})[key] = val

    return overrides


def get_model_info(name: str) -> Optional[Dict[str, str]]:
    """Get information about a model."""
    return AVAILABLE_MODELS.get(name)


def get_all_model_names() -> List[str]:
    """Get list of all registered model names."""
    return list(AVAILABLE_MODELS.keys())
