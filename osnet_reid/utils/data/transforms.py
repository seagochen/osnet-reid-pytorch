"""
Data transforms for ReID / face recognition training.

Configurable augmentations driven by the augmentation config section.
Supports both person ReID (tall rectangular crops) and face recognition (square crops).
"""
from torchvision import transforms

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(height=256, width=128, aug_cfg=None):
    """
    Training transforms with configurable augmentations.

    Args:
        height: Input image height.
        width: Input image width.
        aug_cfg: Augmentation config dict from config['data']['augmentation'].
                 If None, uses defaults suitable for person ReID.
    """
    if aug_cfg is None:
        aug_cfg = {}

    pad = aug_cfg.get('pad', 10)
    flip_p = aug_cfg.get('random_flip', 0.5)
    brightness = aug_cfg.get('brightness', 0.2)
    contrast = aug_cfg.get('contrast', 0.15)
    saturation = aug_cfg.get('saturation', 0.1)
    hue = aug_cfg.get('hue', 0.0)
    erasing_p = aug_cfg.get('random_erasing', 0.5)
    erasing_scale = aug_cfg.get('erasing_scale', [0.02, 0.4])
    rotation = aug_cfg.get('random_rotation', 0)

    t = [
        transforms.Resize((height, width)),
        transforms.RandomHorizontalFlip(p=flip_p),
    ]

    if rotation > 0:
        t.append(transforms.RandomRotation(rotation))

    t.extend([
        transforms.Pad(pad),
        transforms.RandomCrop((height, width)),
        transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=erasing_p, scale=tuple(erasing_scale), value=0),
    ])

    return transforms.Compose(t)


def get_val_transforms(height=256, width=128):
    """Validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
