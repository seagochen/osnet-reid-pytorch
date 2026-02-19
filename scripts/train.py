"""
Training script for OSNet ReID Model.

Usage:
    # Use YAML config file
    python scripts/train.py --config configs/reid.yaml

    # Use YAML + CLI overrides
    python scripts/train.py --config configs/reid.yaml --arch osnet_x0_5 --epochs 80

    # Pure CLI (no YAML)
    python scripts/train.py --data-root /path/to/dataset --csv labels.csv --arch osnet_x1_0

    # List available models
    python scripts/train.py --list-models
"""
import os
import sys
import yaml
import argparse
import torch
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from osnet_reid.training import (
    list_available_models,
    validate_model,
    build_config,
    Trainer,
)
from osnet_reid.utils import init_seeds, colorstr, increment_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train OSNet ReID Model')

    # ===== Config File =====
    parser.add_argument('--config', type=str, default='',
                        help='Path to YAML config file (e.g., configs/reid.yaml)')

    # ===== Model =====
    parser.add_argument('--list-models', action='store_true',
                        help='List all available OSNet models and exit')
    parser.add_argument('--arch', type=str, default='osnet_x1_0',
                        help='OSNet architecture (default: osnet_x1_0)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use pretrained weights')
    parser.add_argument('--num-classes', type=int, default=0,
                        help='Number of identities (0 = auto-detect)')
    parser.add_argument('--reid-dim', type=int, default=512,
                        help='ReID feature dimension (default: 512)')

    # ===== Data =====
    parser.add_argument('--task', type=str, default='reid',
                        choices=['reid', 'face'],
                        help="Task type: 'reid' (person ReID) or 'face' (face recognition)")
    parser.add_argument('--data-root', type=str, default='',
                        help='Dataset root directory')
    parser.add_argument('--csv', type=str, default='labels.csv',
                        help='CSV filename (default: labels.csv)')
    parser.add_argument('--img-height', type=int, default=256,
                        help='Input image height (default: 256)')
    parser.add_argument('--img-width', type=int, default=128,
                        help='Input image width (default: 128)')
    parser.add_argument('--num-instances', type=int, default=4,
                        help='Images per identity in a batch (default: 4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')

    # ===== Training =====
    parser.add_argument('--epochs', type=int, default=60,
                        help='Training epochs (default: 60)')
    parser.add_argument('--lr', type=float, default=3.5e-4,
                        help='Head learning rate (default: 3.5e-4)')
    parser.add_argument('--backbone-lr', type=float, default=3.5e-5,
                        help='Backbone learning rate (default: 3.5e-5)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')
    parser.add_argument('--triplet-margin', type=float, default=0.3,
                        help='Triplet loss margin (default: 0.3)')
    parser.add_argument('--label-smooth', type=float, default=0.1,
                        help='Label smoothing epsilon (default: 0.1)')
    parser.add_argument('--triplet-weight', type=float, default=1.0,
                        help='Triplet loss weight (default: 1.0)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Warmup epochs (default: 5)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')

    # ===== Advanced =====
    parser.add_argument('--ema', action='store_true',
                        help='Use Exponential Moving Average')

    # ===== Output =====
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Save directory (default: runs/train)')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name (default: exp)')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from experiment name')
    parser.add_argument('--device', type=str, default='',
                        help='CUDA device (default: auto)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Handle --list-models
    if args.list_models:
        list_available_models()
        return

    # Build config (YAML + CLI overrides)
    config = build_config(args)

    # Resolve data paths
    data_cfg = config['data']
    data_root = data_cfg['root'] or getattr(args, 'data_root', '')

    if not data_root:
        print(colorstr('bright_red', 'Error: data.root is required'))
        print('Usage:')
        print('  python scripts/train.py --config configs/reid.yaml')
        print('  python scripts/train.py --data-root /path/to/dataset --csv labels.csv')
        return

    # Resolve CSV path
    csv_path = data_cfg['csv']
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(data_root, csv_path)

    data_cfg['root'] = data_root
    data_cfg['csv'] = csv_path

    # Validate model
    arch = config['model']['arch']
    if not validate_model(arch):
        print(colorstr('bright_red', f"Error: Unknown model '{arch}'"))
        print('Use --list-models to see available options')
        return

    # Initialize
    seed = config['advanced']['seed']
    init_seeds(seed)

    # Setup directories
    output_cfg = config['output']
    project = output_cfg['project']
    name = output_cfg['name']

    if args.resume:
        save_dir = Path(project) / args.resume
        if not save_dir.exists():
            print(colorstr('bright_red', f'Experiment not found: {save_dir}'))
            return
    else:
        save_dir = Path(increment_path(Path(project) / name))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    with open(save_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Setup device
    device_str = config['advanced']['device']
    device = torch.device('cuda' if torch.cuda.is_available() and device_str != 'cpu' else 'cpu')

    # Print info
    img_size = config['model']['input_size']
    print(colorstr('bright_green', f'\nStarting training on {device}'))
    print(colorstr('bright_cyan', f'Config: {args.config or "(CLI only)"}'))
    print(colorstr('bright_cyan', f'Architecture: {arch}'))
    print(colorstr('bright_cyan', f'Image size: {img_size[0]}x{img_size[1]}'))
    print(colorstr('bright_cyan', f'ReID dim: {config["model"]["reid_dim"]}'))
    print(colorstr('bright_cyan', f'Epochs: {config["train"]["epochs"]}'))
    print(colorstr('bright_cyan', f'Save dir: {save_dir}'))

    # Create trainer and run
    trainer = Trainer(config, args, save_dir, device)
    trainer.setup_data()
    trainer.setup_model()
    trainer.train()


if __name__ == '__main__':
    main()
