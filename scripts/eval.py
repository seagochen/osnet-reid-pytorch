"""
Evaluation script for OSNet ReID Model.

Loads a trained experiment and computes:
  - EER (Equal Error Rate) and optimal threshold
  - Rank-1 / Rank-5 / Rank-10 / mAP accuracy (CMC evaluation)

Usage:
    # From experiment directory (auto-finds config.yaml + weights/best.pt)
    python scripts/eval.py --exp runs/train/exp3

    # Full evaluation (EER + CMC/mAP)
    python scripts/eval.py --exp runs/train/exp3 --cmc

    # Use final.pt instead of best.pt
    python scripts/eval.py --exp runs/train/exp3 --weights-name final.pt

    # Direct weights path (reads config from checkpoint or sibling config.yaml)
    python scripts/eval.py --weights runs/train/exp3/weights/best.pt

    # Weights + explicit YAML config
    python scripts/eval.py --weights best.pt --config configs/reid.yaml

    # Override dataset
    python scripts/eval.py --exp runs/train/exp3 --data-root /other/data --csv labels.csv
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from osnet_reid.models import ReIDModel
from osnet_reid.utils import ReIDDataset, get_val_transforms, init_seeds
from osnet_reid.training.evaluator import find_best_threshold


def _load_yaml(path):
    """Load YAML config file."""
    p = Path(path)
    if not p.exists():
        print(f"Error: Config file not found: {p}")
        sys.exit(1)
    with open(p, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    print(f"Config: {p}")
    return cfg


def extract_features(model, dataloader, device):
    """Extract L2-normalized features and labels from dataloader."""
    model.eval().to(device)

    all_features = []
    all_pids = []
    all_camids = []

    with torch.no_grad():
        for images, pids, camids in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = model(images, task='reid')
            all_features.append(feats.cpu())
            all_pids.append(pids)
            all_camids.append(camids)

    features = torch.cat(all_features, dim=0)
    pids = torch.cat(all_pids, dim=0)
    camids = torch.cat(all_camids, dim=0)

    return features, pids, camids


def compute_cmc_map(features, pids, camids, ranks=(1, 5, 10)):
    """
    Compute CMC (Cumulative Matching Characteristics) and mAP.

    For each query, rank all other samples by distance and check
    if the correct identity appears within top-k.

    Args:
        features: [N, D] L2-normalized feature tensor
        pids: [N] identity labels
        camids: [N] camera IDs
        ranks: Tuple of rank values to evaluate

    Returns:
        Dict with rank-k accuracies and mAP
    """
    n = features.size(0)

    # Cosine distance matrix (features are already L2-normalized)
    dist_mat = 1 - torch.mm(features, features.t())  # [N, N]

    pids_np = pids.numpy()
    camids_np = camids.numpy()
    dist_np = dist_mat.numpy()

    all_ap = []
    all_cmc = np.zeros(max(ranks))

    num_valid_queries = 0

    for i in range(n):
        query_pid = pids_np[i]

        # Gallery: all except self
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        gallery_dist = dist_np[i][mask]
        gallery_pids = pids_np[mask]

        # Check if there are positive matches in gallery
        matches = (gallery_pids == query_pid)
        if not np.any(matches):
            continue

        num_valid_queries += 1

        # Sort by distance
        indices = np.argsort(gallery_dist)
        sorted_matches = matches[indices]

        # CMC: first correct match position
        first_match = np.argmax(sorted_matches)
        for ri, r in enumerate(ranks):
            if first_match < r:
                all_cmc[ri] += 1

        # AP (Average Precision)
        num_correct = 0
        sum_precision = 0.0
        for j, is_match in enumerate(sorted_matches):
            if is_match:
                num_correct += 1
                sum_precision += num_correct / (j + 1)
        ap = sum_precision / np.sum(matches)
        all_ap.append(ap)

    if num_valid_queries == 0:
        print("Warning: No valid queries found")
        return {f'rank-{r}': 0.0 for r in ranks}, 0.0

    cmc = {f'rank-{r}': all_cmc[ri] / num_valid_queries for ri, r in enumerate(ranks)}
    mAP = float(np.mean(all_ap))

    return cmc, mAP


def resolve_config_and_weights(args):
    """
    Resolve config dict and weights path from args.

    Priority:
      1. --exp: load config.yaml from experiment dir, weights from weights/best.pt
      2. --weights: load config from checkpoint, fallback to sibling config.yaml

    CLI --data-root / --csv override data paths in any mode.

    Returns:
        (config, weights_path)
    """
    config = None
    weights_path = None

    if args.exp:
        exp_dir = Path(args.exp)
        if not exp_dir.exists():
            print(f"Error: Experiment directory not found: {exp_dir}")
            sys.exit(1)

        # Load config.yaml from experiment directory
        config_path = exp_dir / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"Config: {config_path}")
        else:
            print(f"Error: config.yaml not found in {exp_dir}")
            sys.exit(1)

        # Resolve weights path
        weights_name = args.weights_name or 'best.pt'
        weights_path = exp_dir / 'weights' / weights_name
        if not weights_path.exists():
            weights_path = exp_dir / 'weights' / 'final.pt'
            if not weights_path.exists():
                print(f"Error: No weights found in {exp_dir / 'weights'}")
                sys.exit(1)

    elif args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"Error: Weights not found: {weights_path}")
            sys.exit(1)

        # Config source: --config > checkpoint > sibling config.yaml
        if args.config:
            config = _load_yaml(args.config)
        else:
            ckpt_peek = torch.load(weights_path, map_location='cpu', weights_only=False)
            config = ckpt_peek.get('config')

            if config is None:
                exp_dir = weights_path.parent.parent
                config_path = exp_dir / 'config.yaml'
                if config_path.exists():
                    config = _load_yaml(str(config_path))

        if config is None:
            print("Error: No config found. Use --config, --exp, or a checkpoint with embedded config")
            sys.exit(1)

    elif args.config:
        config = _load_yaml(args.config)

        if not args.weights:
            print("Error: --config requires --weights to specify model weights")
            sys.exit(1)

    else:
        print("Error: Provide --exp, --weights, or --config + --weights")
        print("Usage:")
        print("  python scripts/eval.py --exp runs/train/exp3")
        print("  python scripts/eval.py --weights best.pt --config configs/reid.yaml")
        sys.exit(1)

    # CLI overrides for data paths
    if args.data_root:
        config['data']['root'] = args.data_root
    if args.csv:
        config['data']['csv'] = args.csv

    return config, weights_path


def main():
    args = parse_args()

    # Resolve config and weights
    config, weights_path = resolve_config_and_weights(args)

    # Load checkpoint
    print(f"Weights: {weights_path}")
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)

    # Resolve data paths
    data_root = config['data']['root']
    csv_path = config['data']['csv']
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(data_root, csv_path)

    # Setup
    init_seeds(config.get('advanced', {}).get('seed', 0))
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')

    img_h, img_w = config['model']['input_size']
    transform = get_val_transforms(img_h, img_w)

    # Load dataset
    dataset = ReIDDataset(csv_file=csv_path, dataset_root=data_root, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Load model
    config['model']['pretrained'] = False
    state_dict = ckpt.get('model_state_dict', ckpt)

    # Infer num_classes from checkpoint or dataset if config has default 0
    if config['model'].get('num_classes', 0) == 0:
        if 'classifier.weight' in state_dict:
            config['model']['num_classes'] = state_dict['classifier.weight'].shape[0]
        else:
            config['model']['num_classes'] = dataset.num_pids

    model = ReIDModel(config)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"\nModel: {config['model']['arch']}")
    print(f"ReID dim: {config['model']['reid_dim']}")
    print(f"Dataset: {len(dataset)} images, {dataset.num_pids} identities")
    print(f"Device: {device}")

    # ===== EER Evaluation =====
    print(f"\n{'='*60}")
    print("EER Evaluation")
    print(f"{'='*60}")
    threshold, eer = find_best_threshold(model, dataloader, device)

    # ===== CMC / mAP Evaluation =====
    cmc = None
    mAP = None
    if args.cmc:
        print(f"\n{'='*60}")
        print("CMC / mAP Evaluation")
        print(f"{'='*60}")

        features, pids, camids = extract_features(model, dataloader, device)
        cmc, mAP = compute_cmc_map(features, pids, camids)

        print(f"\nResults:")
        for rank_name, acc in cmc.items():
            print(f"  {rank_name}: {acc:.2%}")
        print(f"  mAP:    {mAP:.2%}")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  EER:       {eer:.4%}")
    print(f"  Threshold: {threshold:.4f}")
    if cmc is not None:
        print(f"  Rank-1:    {cmc['rank-1']:.2%}")
        print(f"  mAP:       {mAP:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate OSNet ReID Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval.py --exp runs/train/exp3
  python scripts/eval.py --exp runs/train/exp3 --cmc
  python scripts/eval.py --exp runs/train/exp3 --weights-name final.pt
  python scripts/eval.py --weights runs/train/exp3/weights/best.pt
  python scripts/eval.py --weights best.pt --config configs/reid.yaml
        """,
    )

    # Source
    parser.add_argument('--exp', type=str, default='',
                        help='Experiment directory (auto-loads config.yaml + weights/best.pt)')
    parser.add_argument('--config', type=str, default='',
                        help='Path to YAML config file (for data/model info)')
    parser.add_argument('--weights', type=str, default='',
                        help='Direct path to model weights (.pt)')
    parser.add_argument('--weights-name', type=str, default='',
                        help='Weights filename when using --exp (default: best.pt)')

    # Data overrides
    parser.add_argument('--data-root', type=str, default='',
                        help='Override dataset root directory')
    parser.add_argument('--csv', type=str, default='',
                        help='Override CSV filename')

    # Evaluation options
    parser.add_argument('--cmc', action='store_true',
                        help='Also compute CMC (Rank-1/5/10) and mAP')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for feature extraction (default: 128)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--device', type=str, default='',
                        help='Device (default: auto)')

    return parser.parse_args()


if __name__ == '__main__':
    main()
