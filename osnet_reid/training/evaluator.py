"""
Evaluation utilities for OSNet ReID model.
- Combined loss validation (CE + metric loss)
- EER threshold computation
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


def find_best_threshold(model, dataloader, device):
    """
    Compute EER (Equal Error Rate) threshold for ReID.

    Extracts features from all images in the dataloader, then computes
    pairwise distances between same-identity and different-identity pairs.

    Args:
        model: ReIDModel
        dataloader: DataLoader
        device: torch device

    Returns:
        Tuple of (best_threshold, eer_value)
    """
    from sklearn.metrics import roc_curve

    model.eval().to(device)

    all_features = []
    all_pids = []

    print("Extracting features for EER calculation...")

    with torch.no_grad():
        for images, pids, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = model(images, task='reid')
            all_features.append(feats.cpu())
            all_pids.append(pids)

    all_features = torch.cat(all_features, dim=0)
    all_pids = torch.cat(all_pids, dim=0)

    # Sample pairs for EER computation (limit to avoid OOM)
    n = min(len(all_features), 2000)
    indices = torch.randperm(len(all_features))[:n]
    features = all_features[indices]
    pids = all_pids[indices]

    positive_distances = []
    negative_distances = []

    print(f"Computing pairwise distances for {n} samples...")

    for i in range(n):
        for j in range(i + 1, min(i + 100, n)):
            dist = F.pairwise_distance(
                features[i].unsqueeze(0), features[j].unsqueeze(0)
            ).item()
            if pids[i] == pids[j]:
                positive_distances.append(dist)
            else:
                negative_distances.append(dist)

    if not positive_distances or not negative_distances:
        print("Warning: Not enough pairs for EER computation")
        return 0.5, 1.0

    print(f"Computed distances: {len(positive_distances)} positive pairs, "
          f"{len(negative_distances)} negative pairs")

    # Compute EER
    labels = np.concatenate([
        np.ones(len(positive_distances)),
        np.zeros(len(negative_distances))
    ])
    all_distances = np.concatenate([
        np.array(positive_distances),
        np.array(negative_distances)
    ])

    fpr, tpr, thresholds = roc_curve(labels, -all_distances)
    fnr = 1 - tpr

    idx_eer = np.argmin(np.abs(fpr - fnr))
    best_threshold = -thresholds[idx_eer]
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2.0

    print(f"\nEER Threshold Results:")
    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  FAR: {fpr[idx_eer]:.4f}")
    print(f"  FRR: {fnr[idx_eer]:.4f}")
    print(f"  EER: {eer:.4%}")

    return best_threshold, eer
