"""
Loss functions for ReID training.

- CrossEntropyLabelSmooth: Softmax with label smoothing
- TripletLoss: Triplet loss with batch hard mining
- CircleLoss: Self-paced pair similarity optimization (CVPR 2020)
- ArcFaceLoss: Additive Angular Margin Loss (CVPR 2019)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Smoothed label: q(k) = (1 - epsilon) * y(k) + epsilon / K
    where K is the number of classes.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits [B, num_classes]
            targets: Ground truth labels [B]

        Returns:
            Loss scalar
        """
        log_probs = self.logsoftmax(inputs)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
        loss = (-targets_smooth * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss with batch hard mining.

    For each anchor in the batch, selects:
    - Hardest positive: same identity, maximum distance
    - Hardest negative: different identity, minimum distance

    Loss = max(0, d(anchor, hardest_pos) - d(anchor, hardest_neg) + margin)

    Requires batch to be sampled with RandomIdentitySampler (P identities x K images).
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Feature embeddings [B, D]
            targets: Identity labels [B]

        Returns:
            Loss scalar
        """
        n = inputs.size(0)

        # Pairwise Euclidean distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # Identity mask
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # Hard positive: max distance among same-identity pairs
        dist_ap, dist_an = [], []
        for i in range(n):
            # Hardest positive
            pos_mask = mask[i].clone()
            pos_mask[i] = False  # exclude self
            if pos_mask.any():
                dist_ap.append(dist[i][pos_mask].max().unsqueeze(0))
            else:
                dist_ap.append(dist[i][i].unsqueeze(0))

            # Hardest negative
            neg_mask = ~mask[i]
            if neg_mask.any():
                dist_an.append(dist[i][neg_mask].min().unsqueeze(0))
            else:
                dist_an.append(dist[i][i].unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CircleLoss(nn.Module):
    """
    Circle Loss: A Unified Perspective of Pair Similarity Optimization (CVPR 2020).

    Uses cosine similarity with self-paced weighting:
    - Harder pairs receive larger gradients automatically
    - Each pair has its own adaptive margin via optimal points Op/On

    L = log[1 + sum_neg exp(γ·αn·(sn - Δn)) · sum_pos exp(-γ·αp·(sp - Δp))]

    where:
        sp, sn = cosine similarities for positive/negative pairs
        αp = max(Op - sp, 0),  αn = max(sn - On, 0)  (self-paced weights)
        Op = 1 + m,  On = -m   (optimal similarity points)
        Δp = 1 - m,  Δn = m    (decision boundaries)
        γ = scale factor

    Requires batch to be sampled with RandomIdentitySampler (P identities x K images).
    """

    def __init__(self, margin=0.25, scale=64):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.Op = 1 + margin   # optimal positive similarity
        self.On = -margin      # optimal negative similarity
        self.delta_p = 1 - margin  # positive decision boundary
        self.delta_n = margin      # negative decision boundary

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Feature embeddings [B, D] (will be L2-normalized internally)
            targets: Identity labels [B]

        Returns:
            Loss scalar
        """
        # L2 normalize for cosine similarity
        inputs = F.normalize(inputs, p=2, dim=1)

        # Cosine similarity matrix [B, B]
        sim_mat = torch.matmul(inputs, inputs.t())

        n = inputs.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        losses = []
        for i in range(n):
            pos_mask = mask[i].clone()
            pos_mask[i] = False  # exclude self
            neg_mask = ~mask[i]

            if not pos_mask.any() or not neg_mask.any():
                continue

            sp = sim_mat[i][pos_mask]
            sn = sim_mat[i][neg_mask]

            # Self-paced weights (detached to avoid second-order gradients)
            ap = torch.clamp_min(self.Op - sp.detach(), min=0.)
            an = torch.clamp_min(sn.detach() - self.On, min=0.)

            # Scaled logits
            logit_p = -ap * (sp - self.delta_p) * self.scale
            logit_n = an * (sn - self.delta_n) * self.scale

            loss = F.softplus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
            losses.append(loss)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0., device=inputs.device, requires_grad=True)


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss (CVPR 2019).

    Adds an angular margin penalty to the target class in cosine space,
    producing more discriminative features on the hypersphere.

    L = CE(s * (cos(θ_yi + m), cos(θ_j)), y)

    where:
        θ = angle between L2-normalized feature and weight vectors
        m = additive angular margin (default 0.5)
        s = feature scale (default 64)

    Unlike CE + metric loss combos, ArcFace is a single unified loss.
    It contains its own learnable weight matrix (class centers on the hypersphere).
    """

    def __init__(self, feature_dim, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Class center weights on the hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin terms
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # For numerical stability: threshold where cos(theta+m) is monotonically decreasing
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Feature embeddings [B, D] (will be L2-normalized internally)
            targets: Identity labels [B]

        Returns:
            Loss scalar
        """
        # L2 normalize features and weights
        features = F.normalize(inputs, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # cos(theta) = features @ weight^T
        cosine = F.linear(features, weight)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-12))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical stability: when theta > pi - m, use linear approximation
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply margin only to target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale

        return self.criterion(logits, targets)
