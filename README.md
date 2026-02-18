# OSNet ReID PyTorch

Person Re-Identification (ReID) training framework based on [OSNet](https://arxiv.org/abs/1905.00953) (Omni-Scale Feature Learning), implemented in PyTorch.

## Features

- **OSNet Backbone**: Multi-scale feature extraction via Omni-Scale Blocks with channel attention
- **5 Architecture Variants**: From ultra-light (0.2M params) to standard (2.2M params)
- **Dual Loss Training**: CrossEntropy with label smoothing + configurable metric loss (Triplet or Circle Loss)
- **Batch Hard Mining**: RandomIdentitySampler constructs P-identity x K-image batches
- **YAML + CLI Config**: Flexible configuration with YAML files and CLI overrides
- **ONNX Export**: Production-ready export with L2-normalized feature output

## Project Structure

```
osnet-reid-pytorch/
├── configs/
│   └── default.yaml              # Training configuration template
├── osnet_reid/
│   ├── models/
│   │   ├── osnet.py              # OSNet backbone architecture
│   │   ├── reid_model.py         # ReID wrapper (backbone + BN neck + classifier)
│   │   └── loss.py               # CE, TripletLoss, CircleLoss
│   ├── training/
│   │   ├── config.py             # Config management & model registry
│   │   ├── trainer.py            # Training pipeline
│   │   └── evaluator.py          # Validation & EER threshold
│   └── utils/
│       ├── general.py            # Seeds, colorstr, increment_path
│       ├── callbacks/            # EarlyStopping, ModelEMA, LR scheduler
│       └── data/                 # ReIDDataset, RandomIdentitySampler, transforms
├── scripts/
│   ├── train.py                  # Training entry point
│   └── export_onnx.py           # ONNX export script
└── requirements.txt
```

## Available Models

| Model | Params | Description |
|-------|--------|-------------|
| `osnet_x1_0` | ~2.2M | Standard (default) |
| `osnet_x0_75` | ~1.3M | Smaller |
| `osnet_x0_5` | ~0.6M | Lightweight |
| `osnet_x0_25` | ~0.2M | Ultra-light |
| `osnet_ibn_x1_0` | ~2.2M | With InstanceNorm (cross-domain) |

```bash
python scripts/train.py --list-models
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0

```bash
pip install -r requirements.txt
```

## Dataset Format

Prepare a CSV file with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `img_path` | Yes | Relative path to image (from dataset root) |
| `person_id` | Yes | Identity label (string or integer) |
| `camera_id` | No | Camera ID (defaults to 0) |

Example `labels.csv`:

```csv
img_path,person_id,camera_id
train/0001/img_001.jpg,0001,1
train/0001/img_002.jpg,0001,2
train/0002/img_001.jpg,0002,1
```

## Training

### Using YAML config (recommended)

```bash
python scripts/train.py --config configs/default.yaml
```

### CLI overrides

```bash
python scripts/train.py --config configs/default.yaml \
    --arch osnet_x0_5 \
    --epochs 80 \
    --batch-size 128 \
    --loss-type circle
```

### Pure CLI (no YAML)

```bash
python scripts/train.py \
    --data-root /path/to/dataset \
    --csv labels.csv \
    --arch osnet_x1_0 \
    --epochs 60
```

### Resume training

```bash
python scripts/train.py --config configs/default.yaml --resume exp
```

### Key training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `arch` | `osnet_x1_0` | OSNet variant |
| `reid_dim` | `512` | Feature embedding dimension |
| `input_size` | `[256, 128]` | Input image [height, width] |
| `epochs` | `60` | Training epochs |
| `batch_size` | `64` | Batch size (P x K) |
| `num_instances` | `4` | Images per identity (K) |
| `lr` | `3.5e-4` | Head learning rate |
| `backbone_lr` | `3.5e-5` | Backbone learning rate |
| `loss_type` | `triplet` | Metric loss: `triplet` or `circle` |
| `triplet_margin` | `0.3` | Triplet loss margin |
| `circle_margin` | `0.25` | Circle loss margin |
| `circle_scale` | `64` | Circle loss scale factor |
| `metric_weight` | `1.0` | Weight for metric loss |
| `label_smooth` | `0.1` | Label smoothing epsilon |
| `warmup_epochs` | `5` | Linear warmup epochs |
| `val_split` | `0.2` | Validation split ratio |

### Training outputs

```
runs/train/exp/
├── config.yaml        # Resolved configuration
└── weights/
    ├── best.pt        # Best validation loss
    ├── checkpoint.pt  # Resumable checkpoint (every 5 epochs)
    └── final.pt       # Final model with config
```

## ONNX Export

Export the trained model for deployment. Outputs L2-normalized feature vectors.

```bash
# Basic export
python scripts/export_onnx.py --weights runs/train/exp/weights/best.pt

# With verification
python scripts/export_onnx.py --weights runs/train/exp/weights/best.pt --verify
```

**ONNX model I/O:**
- Input: `input` — `[batch_size, 3, 256, 128]`
- Output: `reid_features` — `[batch_size, 512]` (L2-normalized)

## Architecture Overview

### ReID Model Pipeline

```
Image [B, 3, H, W]
  → OSNet Backbone → GAP → [B, feature_dim]
  → BN Neck (BatchNorm1d)
  → Features (for metric loss)  +  Classifier (for CE loss)
```

### OSNet Block (OSBlock)

The core innovation of OSNet is the Omni-Scale Block, which captures multi-scale features through parallel streams:

```
Input
  ├── LightConv3x3 x1  (receptive field: 3x3)
  ├── LightConv3x3 x2  (receptive field: 5x5)
  ├── LightConv3x3 x3  (receptive field: 7x7)
  └── LightConv3x3 x4  (receptive field: 9x9)
  → ChannelGate attention fusion
  → Residual connection
  → Output
```

Each `LightConv3x3` uses depthwise separable convolutions for efficiency.

## References

- [Omni-Scale Feature Learning for Person Re-Identification (ICCV 2019)](https://arxiv.org/abs/1905.00953)
- [Circle Loss: A Unified Perspective of Pair Similarity Optimization (CVPR 2020)](https://arxiv.org/abs/2002.10857)

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
