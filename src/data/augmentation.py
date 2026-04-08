"""Augmentation pipeline using albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Computed from 787 real HPA IF images (proteinatlas.org)
# Channel order: red (microtubule), green (protein), blue (nucleus), yellow (ER)
HPA_MEAN = [0.0321, 0.0408, 0.0078, 0.0814]
HPA_STD = [0.0527, 0.0746, 0.0201, 0.1399]


def get_train_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3,
        ),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
        A.Normalize(
            mean=HPA_MEAN,
            std=HPA_STD,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=HPA_MEAN,
            std=HPA_STD,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int = 256) -> list[A.Compose]:
    """Return a list of deterministic augmentation pipelines for test-time augmentation."""
    base_norm = [
        A.Normalize(mean=HPA_MEAN, std=HPA_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return [
        # identity
        A.Compose([A.Resize(image_size, image_size)] + base_norm),
        # horizontal flip
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0)] + base_norm),
        # vertical flip
        A.Compose([A.Resize(image_size, image_size), A.VerticalFlip(p=1.0)] + base_norm),
        # 90-degree rotation (deterministic via Transpose + HorizontalFlip)
        A.Compose([A.Resize(image_size, image_size), A.Transpose(p=1.0), A.HorizontalFlip(p=1.0)] + base_norm),
    ]
