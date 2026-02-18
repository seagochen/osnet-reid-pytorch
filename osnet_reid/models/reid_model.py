"""
ReID Model: OSNet backbone + BN neck + Classifier.

Architecture:
    Input Image (H x W, default 256x128)
          |
    OSNet Backbone
          |
    GAP -> [B, feature_dim]
          |
    BN Neck (BatchNorm1d)
          |
       +--+--+
       |     |
    Feature  Classifier
    [B, D]   [B, num_pids]
    (triplet) (CE loss)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .osnet import build_osnet


class ReIDModel(nn.Module):
    """
    ReID model with shared OSNet backbone and task-specific outputs.

    The OSNet backbone already includes GAP and FC layers. This wrapper adds:
    - BN neck for feature normalization
    - Classifier head for identity classification

    Training: returns (features_before_bn, logits) for triplet + CE loss
    Inference: returns L2-normalized features for matching
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary with keys:
                - model.arch: OSNet variant name (default: 'osnet_x1_0')
                - model.pretrained: bool or pretrained weight path
                - model.num_classes: number of identities
                - model.reid_dim: feature dimension (default: 512)
                - model.input_size: [height, width] (default: [256, 128])
        """
        super().__init__()

        model_cfg = config['model']
        arch = model_cfg.get('arch', 'osnet_x1_0')
        pretrained = model_cfg.get('pretrained', True)
        num_classes = model_cfg.get('num_classes', 0)
        reid_dim = model_cfg.get('reid_dim', 512)

        pretrained_path = pretrained if isinstance(pretrained, str) else ''

        # Build OSNet backbone (without internal classifier)
        self.backbone = build_osnet(arch, num_classes=0, pretrained_path=pretrained_path,
                                    feature_dim=reid_dim)
        backbone_dim = self.backbone.feature_dim

        # BN Neck
        self.bottleneck = nn.BatchNorm1d(backbone_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        # Classifier
        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(backbone_dim, num_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)

        self.backbone_dim = backbone_dim
        self.reid_dim = reid_dim
        self.num_classes = num_classes

    def forward(self, x, task='both'):
        """
        Forward pass.

        Args:
            x: Input image [B, 3, H, W]
            task: 'reid' | 'classification' | 'both'

        Returns:
            task='reid': L2-normalized features [B, feature_dim]
            task='classification': logits [B, num_classes]
            task='both': (features, logits) where features are pre-BN
        """
        # Backbone forward: get features after GAP + FC
        feat = self.backbone(x)  # [B, backbone_dim]

        if task == 'reid':
            feat_bn = self.bottleneck(feat)
            return F.normalize(feat_bn, p=2, dim=1)

        elif task == 'classification':
            feat_bn = self.bottleneck(feat)
            if self.classifier is not None:
                return self.classifier(feat_bn)
            return feat_bn

        else:  # 'both' - training mode
            feat_bn = self.bottleneck(feat)
            if self.classifier is not None:
                logits = self.classifier(feat_bn)
                # Return: pre-BN features (for triplet), logits (for CE)
                return feat, logits
            return feat, feat_bn

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")


if __name__ == '__main__':
    config = {
        'model': {
            'arch': 'osnet_x1_0',
            'pretrained': False,
            'num_classes': 751,
            'reid_dim': 512,
            'input_size': [256, 128],
        }
    }

    model = ReIDModel(config)
    model.train()
    dummy = torch.randn(4, 3, 256, 128)

    # Training mode
    feat, logits = model(dummy, task='both')
    print(f"Training mode:")
    print(f"  Features: {feat.shape}")
    print(f"  Logits: {logits.shape}")

    # Inference mode
    model.eval()
    with torch.no_grad():
        reid_feat = model(dummy, task='reid')
        print(f"\nInference mode:")
        print(f"  ReID features: {reid_feat.shape}")
        print(f"  L2 norm check: {torch.norm(reid_feat[0]).item():.4f}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nTotal parameters: {params:.2f}M")
