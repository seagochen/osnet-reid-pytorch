"""
OSNet: Omni-Scale Feature Learning for Person Re-Identification.

Reference:
    Zhou et al. "Omni-Scale Feature Learning for Person Re-Identification" (ICCV 2019)
    https://arxiv.org/abs/1905.00953

Ported from torchreid: https://github.com/KaiyangZhou/deep-person-reid
Pure PyTorch implementation, no external dependencies.

Supported variants:
    - osnet_x1_0:     [64, 256, 384, 512], ~2.2M params
    - osnet_x0_75:    [48, 192, 288, 384], ~1.3M params
    - osnet_x0_5:     [32, 128, 192, 256], ~0.6M params
    - osnet_x0_25:    [16,  64,  96, 128], ~0.2M params
    - osnet_ibn_x1_0: [64, 256, 384, 512] + InstanceNorm, ~2.2M params
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# Building blocks
# ============================================================

class ConvLayer(nn.Module):
    """Conv2d + BN (or IN) + ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False, groups=groups,
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 conv + BN + ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 conv + BN (no ReLU)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class Conv3x3(nn.Module):
    """3x3 conv + BN + ReLU."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3,
            stride=stride, padding=1, bias=False, groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LightConv3x3(nn.Module):
    """
    Lightweight 3x3 convolution (depthwise separable).

    1x1 pointwise (channels unchanged) -> 3x3 depthwise.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            stride=1, padding=1, bias=False, groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(self.bn(x))


# ============================================================
# Aggregation gate (channel attention)
# ============================================================

class ChannelGate(nn.Module):
    """
    Channel attention gate for multi-scale feature aggregation.

    GAP -> FC -> ReLU -> FC -> Sigmoid
    """

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16):
        super().__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels, in_channels // reduction, 1, bias=True, padding=0,
        )
        self.norm1 = nn.LayerNorm([in_channels // reduction, 1, 1])
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction, num_gates, 1, bias=True, padding=0,
        )

        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(f"Unknown gate activation: {gate_activation}")

    def forward(self, x):
        inp = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return inp * x


# ============================================================
# OSBlock: Omni-Scale residual block
# ============================================================

class OSBlock(nn.Module):
    """
    Omni-Scale building block.

    4 parallel multi-scale streams with different receptive fields:
      - Stream 1: 1x LightConv3x3 (RF 3x3)
      - Stream 2: 2x LightConv3x3 (RF 5x5)
      - Stream 3: 3x LightConv3x3 (RF 7x7)
      - Stream 4: 4x LightConv3x3 (RF 9x9)

    Each stream output is gated by ChannelGate then summed.
    """

    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4, **kwargs):
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )

        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x

        x1 = self.conv1(x)

        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)

        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)

        x3 = self.conv3(x2)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = x3 + identity

        if self.IN is not None:
            out = self.IN(out)

        return F.relu(out)


# ============================================================
# Network architecture
# ============================================================

class OSNet(nn.Module):
    """
    Omni-Scale Network.

    Architecture:
        conv1 (stem): 7x7 conv + MaxPool
        conv2: N x OSBlock
        transition1: Conv1x1 + AvgPool
        conv3: N x OSBlock
        transition2: Conv1x1 + AvgPool
        conv4: N x OSBlock
        conv5: 1x1 conv
        GAP
        fc (classifier, optional)

    Args:
        num_classes: Number of identities (0 = feature extractor only)
        blocks: List of block counts [conv2, conv3, conv4]
        channels: List of channel widths [conv2, conv3, conv4, conv5]
        feature_dim: Output feature dimension (default: 512)
        loss: Not used, kept for API compatibility
        IN: Use InstanceNorm in OSBlock
    """

    def __init__(self, num_classes, blocks, channels, feature_dim=512,
                 loss='softmax', IN=False, **kwargs):
        super().__init__()
        num_blocks = len(blocks)
        assert num_blocks == 3

        self.feature_dim = feature_dim

        # conv1: stem
        self.conv1 = nn.Sequential(
            ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # conv2
        self.conv2 = self._make_layer(OSBlock, blocks[0], channels[0], channels[1], IN=IN)

        # transition1
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2),
        )

        # conv3
        self.conv3 = self._make_layer(OSBlock, blocks[1], channels[1], channels[2], IN=IN)

        # transition2
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2),
        )

        # conv4
        self.conv4 = self._make_layer(OSBlock, blocks[2], channels[2], channels[3], IN=IN)

        # conv5: final 1x1 refinement
        self.conv5 = Conv1x1(channels[3], channels[3])

        # Global average pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # FC layer
        self.fc = self._construct_fc(self.feature_dim, channels[3], num_classes)

        self._init_params()

    def _construct_fc(self, feature_dim, input_dim, num_classes):
        """Build FC layers for feature + optional classifier."""
        fc = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes) if num_classes > 0 else None
        return fc

    def _make_layer(self, block, num_blocks, in_channels, out_channels, IN=False):
        layers = []
        layers.append(block(in_channels, out_channels, IN=IN))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, IN=IN))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        """Extract feature maps before pooling."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        """
        Forward pass.

        Args:
            x: Input image [B, 3, H, W]
            return_featuremaps: If True, return feature maps before pooling

        Returns:
            If training and classifier exists: logits [B, num_classes]
            Otherwise: features [B, feature_dim]
        """
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        v = self.fc(v)

        if not self.training:
            return v

        if self.classifier is not None:
            y = self.classifier(v)
            return v, y

        return v


# ============================================================
# Factory functions
# ============================================================

def _osnet(num_classes, blocks, channels, feature_dim=512, IN=False, pretrained_path='', **kwargs):
    """Generic OSNet constructor."""
    model = OSNet(num_classes, blocks, channels, feature_dim=feature_dim, IN=IN, **kwargs)
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")
    return model


def osnet_x1_0(num_classes=0, pretrained_path='', **kwargs):
    """OSNet x1.0: ~2.2M params, channels [64, 256, 384, 512]."""
    return _osnet(num_classes, [2, 2, 2], [64, 256, 384, 512],
                  pretrained_path=pretrained_path, **kwargs)


def osnet_x0_75(num_classes=0, pretrained_path='', **kwargs):
    """OSNet x0.75: ~1.3M params, channels [48, 192, 288, 384]."""
    return _osnet(num_classes, [2, 2, 2], [48, 192, 288, 384],
                  pretrained_path=pretrained_path, **kwargs)


def osnet_x0_5(num_classes=0, pretrained_path='', **kwargs):
    """OSNet x0.5: ~0.6M params, channels [32, 128, 192, 256]."""
    return _osnet(num_classes, [2, 2, 2], [32, 128, 192, 256],
                  pretrained_path=pretrained_path, **kwargs)


def osnet_x0_25(num_classes=0, pretrained_path='', **kwargs):
    """OSNet x0.25: ~0.2M params, channels [16, 64, 96, 128]."""
    return _osnet(num_classes, [2, 2, 2], [16, 64, 96, 128],
                  pretrained_path=pretrained_path, **kwargs)


def osnet_ibn_x1_0(num_classes=0, pretrained_path='', **kwargs):
    """OSNet x1.0 with InstanceNorm: ~2.2M params, better cross-domain generalization."""
    return _osnet(num_classes, [2, 2, 2], [64, 256, 384, 512],
                  IN=True, pretrained_path=pretrained_path, **kwargs)


# Model registry
OSNET_MODELS = {
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_x0_5': osnet_x0_5,
    'osnet_x0_25': osnet_x0_25,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
}


def build_osnet(arch, num_classes=0, pretrained_path='', **kwargs):
    """
    Build OSNet by architecture name.

    Args:
        arch: Architecture name (e.g., 'osnet_x1_0')
        num_classes: Number of identity classes (0 = feature extractor only)
        pretrained_path: Path to pretrained weights

    Returns:
        OSNet model
    """
    if arch not in OSNET_MODELS:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list(OSNET_MODELS.keys())}")
    return OSNET_MODELS[arch](num_classes=num_classes, pretrained_path=pretrained_path, **kwargs)


if __name__ == '__main__':
    # Test all variants
    for name, fn in OSNET_MODELS.items():
        print(f"\nTesting {name}:")
        model = fn(num_classes=751)
        model.eval()
        x = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            out = model(x)
        print(f"  Input: {x.shape}")
        print(f"  Output: {out.shape}")
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Parameters: {params:.2f}M")
