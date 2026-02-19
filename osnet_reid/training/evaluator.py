"""
Evaluation utilities for OSNet ReID model.
- Combined loss validation (CE + metric loss)
- EER threshold computation
- TAR@FAR verification metrics
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def validate(model, val_loader, criterion_ce, criterion_metric, metric_weight, device):
    """
    Validate ReID model on combined CE + metric loss.

    Args:
        model: ReIDModel
        val_loader: DataLoader
        criterion_ce: CrossEntropyLabelSmooth
        criterion_metric: TripletLoss or CircleLoss
        metric_weight: Weight for metric loss
        device: torch device

    Returns:
        Tuple of (total_loss, ce_loss, metric_loss)
    """
    model.eval()
    total_ce = 0
    total_metric = 0
    batch_count = 0

    with torch.no_grad():
        for images, pids, _ in val_loader:
            images = images.to(device)
            pids = pids.to(device)

            features, logits = model(images, task='both')

            loss_ce = criterion_ce(logits, pids)
            loss_metric = criterion_metric(features, pids)

            total_ce += loss_ce.item()
            total_metric += loss_metric.item()
            batch_count += 1

    avg_ce = total_ce / max(batch_count, 1)
    avg_metric = total_metric / max(batch_count, 1)
    avg_total = avg_ce + metric_weight * avg_metric

    return avg_total, avg_ce, avg_metric


def extract_all_features(model, dataloader, device):
    """Extract L2-normalized features and labels from dataloader."""
    model.eval().to(device)

    all_features = []
    all_pids = []

    with torch.no_grad():
        for images, pids, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = model(images, task='reid')
            all_features.append(feats.cpu())
            all_pids.append(pids)

    return torch.cat(all_features, dim=0), torch.cat(all_pids, dim=0)


def _compute_pair_distances(features, pids, n_samples=2000):
    """
    Sample pairs and compute pairwise Euclidean distances.

    Returns (positive_distances, negative_distances) as numpy arrays.
    """
    n = min(len(features), n_samples)
    indices = torch.randperm(len(features))[:n]
    features = features[indices]
    pids = pids[indices]

    positive_distances = []
    negative_distances = []

    for i in range(n):
        for j in range(i + 1, min(i + 100, n)):
            dist = F.pairwise_distance(
                features[i].unsqueeze(0), features[j].unsqueeze(0)
            ).item()
            if pids[i] == pids[j]:
                positive_distances.append(dist)
            else:
                negative_distances.append(dist)

    return np.array(positive_distances), np.array(negative_distances)


def find_best_threshold(model, dataloader, device):
    """
    Compute EER (Equal Error Rate) and TAR@FAR verification metrics.

    Extracts features from all images in the dataloader, then computes
    pairwise distances between same-identity and different-identity pairs.

    Args:
        model: ReIDModel
        dataloader: DataLoader
        device: torch device

    Returns:
        Dict with keys: threshold, eer, tar_at_far (dict of FAR->TAR)
    """
    from sklearn.metrics import roc_curve

    print("Extracting features for EER calculation...")
    all_features, all_pids = extract_all_features(model, dataloader, device)

    print(f"Computing pairwise distances for {min(len(all_features), 2000)} samples...")
    pos_dist, neg_dist = _compute_pair_distances(all_features, all_pids)

    if len(pos_dist) == 0 or len(neg_dist) == 0:
        print("Warning: Not enough pairs for EER computation")
        return {'threshold': 0.5, 'eer': 1.0, 'tar_at_far': {}}

    print(f"Computed distances: {len(pos_dist)} positive pairs, "
          f"{len(neg_dist)} negative pairs")

    # Build ROC curve (negate distances so higher = more similar)
    labels = np.concatenate([np.ones(len(pos_dist)), np.zeros(len(neg_dist))])
    distances = np.concatenate([pos_dist, neg_dist])
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    fnr = 1 - tpr

    # EER: where FPR ≈ FNR
    idx_eer = np.argmin(np.abs(fpr - fnr))
    best_threshold = -thresholds[idx_eer]
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2.0

    # TAR@FAR: interpolate TPR at specific FAR thresholds
    tar_at_far = {}
    for target_far in [1e-1, 1e-2, 1e-3]:
        # Find the largest TPR where FPR <= target_far
        valid = fpr <= target_far
        if np.any(valid):
            tar_at_far[target_far] = float(tpr[valid].max())
        else:
            tar_at_far[target_far] = 0.0

    print(f"\nEER Threshold Results:")
    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  FAR: {fpr[idx_eer]:.4f}")
    print(f"  FRR: {fnr[idx_eer]:.4f}")
    print(f"  EER: {eer:.4%}")
    print(f"\nVerification (TAR@FAR):")
    for far_val, tar_val in sorted(tar_at_far.items(), reverse=True):
        print(f"  TAR@FAR={far_val:.0e}: {tar_val:.4%}")

    return {'threshold': best_threshold, 'eer': eer, 'tar_at_far': tar_at_far}
