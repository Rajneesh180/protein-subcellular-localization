"""Augmentation pipeline using albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 512) -> A.Compose:
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
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.Normalize(
            mean=[0.08] * 4,  # HPA images are sparse/dark
            std=[0.15] * 4,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.08] * 4,
            std=[0.15] * 4,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
