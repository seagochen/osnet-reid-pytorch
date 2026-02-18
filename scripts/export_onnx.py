"""
ONNX export script for OSNet ReID Model.

Exports feature extraction only (backbone + BN neck, no classifier).
Output: L2-normalized feature vector [B, feature_dim].

Usage:
    python scripts/export_onnx.py --weights runs/train/exp/weights/best.pt
    python scripts/export_onnx.py --weights runs/train/exp/weights/best.pt --verify
"""
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from osnet_reid.models import ReIDModel


class ONNXExportWrapper(nn.Module):
    """
    ONNX-friendly wrapper for ReID feature extraction.

    Output: L2-normalized feature vector [B, feature_dim]
    """

    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model.backbone
        self.bottleneck = base_model.bottleneck

    def forward(self, x):
        # Backbone: extract features
        feat = self.backbone(x)  # [B, feature_dim]
        # BN Neck
        feat = self.bottleneck(feat)
        # L2 normalize
        feat = F.normalize(feat, p=2, dim=1)
        return feat


def export_onnx(args):
    """Export model to ONNX format."""
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights not found: {weights_path}")
        return

    # Load checkpoint
    print(f"Loading weights: {weights_path}")
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)

    # Get config from checkpoint or use defaults
    config = ckpt.get('config', {
        'model': {
            'arch': args.arch,
            'pretrained': False,
            'num_classes': args.num_classes,
            'reid_dim': args.reid_dim,
            'input_size': [args.img_height, args.img_width],
        }
    })
    config['model']['pretrained'] = False
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))

    # Infer num_classes from checkpoint if needed
    if config['model'].get('num_classes', 0) == 0 or 'classifier.weight' in state_dict:
        if 'classifier.weight' in state_dict:
            config['model']['num_classes'] = state_dict['classifier.weight'].shape[0]

    # Create model and load weights
    model = ReIDModel(config)
    model.load_state_dict(state_dict)
    model.eval()

    # Wrap for ONNX (feature extraction only)
    export_model = ONNXExportWrapper(model).eval()

    # Dummy input
    img_h = config['model']['input_size'][0]
    img_w = config['model']['input_size'][1]
    dummy_input = torch.randn(1, 3, img_h, img_w)

    # Output path
    output_path = weights_path.parent / f'{weights_path.stem}.onnx'
    if args.output:
        output_path = Path(args.output)

    # Export
    feature_dim = config['model'].get('reid_dim', 512)
    print(f"\nExporting to: {output_path}")

    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['reid_features'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'reid_features': {0: 'batch_size'},
            },
            verbose=False,
        )

    print(f"\nONNX export complete: {output_path}")
    print(f"  Input:  input [{img_h}x{img_w}]")
    print(f"  Output: reid_features [B, {feature_dim}]")

    # Verify
    if args.verify:
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np

            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("\nONNX model structure check: PASSED")

            # Compare outputs
            session = ort.InferenceSession(str(output_path))
            test_input = np.random.randn(1, 3, img_h, img_w).astype(np.float32)
            onnx_outputs = session.run(None, {'input': test_input})

            torch_input = torch.from_numpy(test_input)
            with torch.no_grad():
                torch_output = export_model(torch_input)

            diff = np.abs(onnx_outputs[0] - torch_output.numpy()).max()
            status = 'PASS' if diff < 1e-5 else f'DIFF={diff:.6f}'
            print(f"  reid_features: {status}")

        except ImportError:
            print("\nSkipping verification (install onnx and onnxruntime)")


def parse_args():
    parser = argparse.ArgumentParser(description='Export OSNet ReID to ONNX')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights (.pt)')
    parser.add_argument('--output', type=str, default='',
                        help='Output ONNX file path')
    parser.add_argument('--img-height', type=int, default=256,
                        help='Input image height (default: 256)')
    parser.add_argument('--img-width', type=int, default=128,
                        help='Input image width (default: 128)')
    parser.add_argument('--verify', action='store_true', default=True,
                        help='Verify ONNX output against PyTorch')
    parser.add_argument('--arch', type=str, default='osnet_x1_0',
                        help='OSNet architecture (used if config not in checkpoint)')
    parser.add_argument('--num-classes', type=int, default=751,
                        help='Number of classes (used if config not in checkpoint)')
    parser.add_argument('--reid-dim', type=int, default=512,
                        help='ReID dimension (used if config not in checkpoint)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    export_onnx(args)
