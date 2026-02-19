"""
Inference script for person ReID and face recognition.

Supports both PyTorch checkpoints and ONNX models.
Provides 1:1 verification and 1:N identification with a gallery.

Usage:
    # 1:1 Verification — are these two images the same person?
    python scripts/inference.py verify --weights best.pt --img1 a.jpg --img2 b.jpg

    # Build gallery and identify
    python scripts/inference.py identify --weights best.pt \\
        --gallery-dir /path/to/gallery/ --query query.jpg

    # Use ONNX model
    python scripts/inference.py verify --onnx model.onnx --img1 a.jpg --img2 b.jpg

Gallery directory structure:
    gallery/
      alice/
        001.jpg
        002.jpg
      bob/
        001.jpg
"""
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union, Tuple, List, Dict

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from osnet_reid.models import ReIDModel
from osnet_reid.utils.data.transforms import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD


class RecognitionSystem:
    """
    Recognition system for person ReID / face verification & identification.

    Supports PyTorch checkpoints and ONNX models.
    Uses cosine similarity on L2-normalized embeddings.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.transform = None
        self._model = None       # PyTorch model
        self._onnx_session = None  # ONNX session
        self.device = torch.device('cpu')
        # Gallery: {name: [D] mean embedding tensor or numpy array}
        self._gallery: Dict[str, np.ndarray] = {}

    @classmethod
    def from_checkpoint(cls, weights_path: str, device: str = '',
                        threshold: float = 0.5) -> 'RecognitionSystem':
        """Load from a PyTorch checkpoint (.pt)."""
        import yaml

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dev = torch.device(device)

        ckpt = torch.load(weights_path, map_location=dev, weights_only=False)

        config = ckpt.get('config')
        if config is None:
            config_path = weights_path.parent.parent / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                raise RuntimeError(f"No config found. Place config.yaml at {config_path}")

        config['model']['pretrained'] = False
        state_dict = ckpt.get('model_state_dict', ckpt)

        if config['model'].get('num_classes', 0) == 0:
            if 'classifier.weight' in state_dict:
                config['model']['num_classes'] = state_dict['classifier.weight'].shape[0]

        model = ReIDModel(config)
        model.load_state_dict(state_dict)
        model.eval().to(dev)

        img_h, img_w = config['model']['input_size']
        system = cls(threshold=threshold)
        system._model = model
        system.device = dev
        system.transform = get_val_transforms(img_h, img_w)
        return system

    @classmethod
    def from_onnx(cls, onnx_path: str, input_size=(112, 112),
                  threshold: float = 0.5) -> 'RecognitionSystem':
        """Load from an ONNX model."""
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        session = ort.InferenceSession(str(onnx_path))

        # Infer input size from model
        input_shape = session.get_inputs()[0].shape  # [B, 3, H, W]
        if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            input_size = (input_shape[2], input_shape[3])

        system = cls(threshold=threshold)
        system._onnx_session = session
        system.transform = get_val_transforms(input_size[0], input_size[1])
        return system

    def _preprocess(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """Load image and return preprocessed numpy array [1, 3, H, W]."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)  # [1, 3, H, W]
        return tensor.numpy()

    def encode(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Encode an image into an L2-normalized embedding vector.

        Returns:
            numpy array of shape [D].
        """
        if self._onnx_session is not None:
            input_np = self._preprocess(image)
            input_name = self._onnx_session.get_inputs()[0].name
            outputs = self._onnx_session.run(None, {input_name: input_np})
            feat = outputs[0][0]  # [D]
            # Normalize (ONNX model should already output normalized, but ensure)
            feat = feat / (np.linalg.norm(feat) + 1e-12)
            return feat
        else:
            input_np = self._preprocess(image)
            tensor = torch.from_numpy(input_np).to(self.device)
            with torch.no_grad():
                feat = self._model(tensor, task='reid')  # [1, D]
            return feat[0].cpu().numpy()

    def verify(self, image1, image2) -> Tuple[bool, float]:
        """
        1:1 verification — are these two images the same person?

        Returns:
            (is_same, similarity): bool and cosine similarity score.
        """
        feat1 = self.encode(image1)
        feat2 = self.encode(image2)
        similarity = float(np.dot(feat1, feat2))
        return similarity >= self.threshold, similarity

    def add(self, name: str, image) -> None:
        """Add an image to the gallery. Multiple images per identity are averaged."""
        feat = self.encode(image)  # [D]
        if name in self._gallery:
            # Running mean: accumulate and re-normalize
            old = self._gallery[name]
            combined = old + feat
            self._gallery[name] = combined / (np.linalg.norm(combined) + 1e-12)
        else:
            self._gallery[name] = feat

    def identify(self, image, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        1:N identification — find the closest identity in the gallery.

        Returns:
            List of (name, similarity) tuples sorted by similarity descending.
            Returns [("unknown", best_score)] if no match exceeds threshold.
        """
        if not self._gallery:
            raise RuntimeError("Gallery is empty. Use add() to enroll identities first.")

        query = self.encode(image)

        results = []
        for name, gallery_feat in self._gallery.items():
            sim = float(np.dot(query, gallery_feat))
            results.append((name, sim))

        results.sort(key=lambda x: x[1], reverse=True)

        if results[0][1] < self.threshold:
            return [("unknown", results[0][1])]

        return results[:top_k]

    @property
    def gallery_names(self) -> List[str]:
        return list(self._gallery.keys())


def build_gallery_from_dir(system: RecognitionSystem, gallery_dir: str) -> None:
    """
    Build gallery from a directory structure:
        gallery_dir/
          person_name/
            img1.jpg
            img2.jpg
    """
    gallery_dir = Path(gallery_dir)
    if not gallery_dir.is_dir():
        print(f"Error: Gallery directory not found: {gallery_dir}")
        sys.exit(1)

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    count = 0
    for person_dir in sorted(gallery_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() in img_exts:
                system.add(name, str(img_path))
                count += 1

    print(f"Gallery: {system.gallery_size} identities, {count} images")


# Alias for gallery_size property if needed
RecognitionSystem.gallery_size = property(lambda self: len(self._gallery))


def cmd_verify(args):
    """Handle verify subcommand."""
    system = _load_system(args)

    is_same, similarity = system.verify(args.img1, args.img2)
    status = "SAME person" if is_same else "DIFFERENT person"

    print(f"\nVerification Result:")
    print(f"  Image 1:    {args.img1}")
    print(f"  Image 2:    {args.img2}")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Result:     {status}")


def cmd_identify(args):
    """Handle identify subcommand."""
    system = _load_system(args)

    # Build gallery
    build_gallery_from_dir(system, args.gallery_dir)

    # Identify query
    results = system.identify(args.query, top_k=args.top_k)

    print(f"\nIdentification Result:")
    print(f"  Query:     {args.query}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Top-{args.top_k}:")
    for rank, (name, sim) in enumerate(results, 1):
        print(f"    #{rank}: {name} (similarity: {sim:.4f})")


def _load_system(args) -> RecognitionSystem:
    """Create RecognitionSystem from CLI args."""
    if args.onnx:
        print(f"Loading ONNX model: {args.onnx}")
        return RecognitionSystem.from_onnx(args.onnx, threshold=args.threshold)
    elif args.weights:
        print(f"Loading checkpoint: {args.weights}")
        return RecognitionSystem.from_checkpoint(
            args.weights, device=args.device, threshold=args.threshold)
    else:
        print("Error: Provide --weights or --onnx")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Person ReID / Face Recognition Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Common args
    parser.add_argument('--weights', type=str, default='',
                        help='Path to PyTorch checkpoint (.pt)')
    parser.add_argument('--onnx', type=str, default='',
                        help='Path to ONNX model')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Cosine similarity threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='',
                        help='Device (default: auto)')

    sub = parser.add_subparsers(dest='command', help='Command')

    # verify
    p_verify = sub.add_parser('verify', help='1:1 verification')
    p_verify.add_argument('--img1', type=str, required=True, help='First image')
    p_verify.add_argument('--img2', type=str, required=True, help='Second image')

    # identify
    p_identify = sub.add_parser('identify', help='1:N identification')
    p_identify.add_argument('--gallery-dir', type=str, required=True,
                            help='Gallery directory (subfolders = identities)')
    p_identify.add_argument('--query', type=str, required=True, help='Query image')
    p_identify.add_argument('--top-k', type=int, default=5, help='Top-K results')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.command == 'verify':
        cmd_verify(args)
    elif args.command == 'identify':
        cmd_identify(args)
    else:
        print("Usage:")
        print("  python scripts/inference.py --weights best.pt verify --img1 a.jpg --img2 b.jpg")
        print("  python scripts/inference.py --weights best.pt identify --gallery-dir gallery/ --query q.jpg")
        print("  python scripts/inference.py --onnx model.onnx verify --img1 a.jpg --img2 b.jpg")
