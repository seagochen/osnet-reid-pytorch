"""
Data transforms for ReID training.

ReID-specific augmentations including random erasing.
"""
from torchvision import transforms

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(height=256, width=128):
    """
    Training transforms with ReID-specific augmentations.

    Resize -> HFlip -> Pad+RandomCrop -> ColorJitter -> ToTensor -> Normalize -> RandomErasing
    """
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(10),
        transforms.RandomCrop((height, width)),
        transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), value=0),
    ])


def get_val_transforms(height=256, width=128):
    """Validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
