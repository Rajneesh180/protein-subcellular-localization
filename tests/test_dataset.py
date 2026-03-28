"""Tests for dataset and data pipeline."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.dataset import HPADataset, HPA_LABELS, NUM_CLASSES, get_label_distribution
from src.data.augmentation import get_train_transforms, get_val_transforms


@pytest.fixture
def dummy_data():
    """Create a minimal dummy HPA dataset for testing."""
    tmpdir = tempfile.mkdtemp()

    # create dummy images
    ids = ["img_001", "img_002", "img_003"]
    colors = ["red", "green", "blue", "yellow"]
    for img_id in ids:
        for color in colors:
            img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
            Image.fromarray(img).save(os.path.join(tmpdir, f"{img_id}_{color}.png"))

    # create dummy CSV
    csv_path = os.path.join(tmpdir, "train.csv")
    df = pd.DataFrame({
        "Id": ids,
        "Target": ["0 5 16", "1 2", "0"],
    })
    df.to_csv(csv_path, index=False)

    return tmpdir, csv_path


def test_dataset_loading(dummy_data):
    tmpdir, csv_path = dummy_data
    dataset = HPADataset(csv_path, tmpdir, image_size=64)

    assert len(dataset) == 3

    image, target = dataset[0]
    assert image.shape == (4, 64, 64)
    assert target.shape == (NUM_CLASSES,)
    assert target[0] == 1.0  # class 0
    assert target[5] == 1.0  # class 5
    assert target[16] == 1.0  # class 16


def test_target_encoding(dummy_data):
    tmpdir, csv_path = dummy_data
    dataset = HPADataset(csv_path, tmpdir, image_size=64)

    # second sample has labels 1, 2
    _, target = dataset[1]
    assert target[1] == 1.0
    assert target[2] == 1.0
    assert target.sum() == 2.0


def test_augmentation_shapes(dummy_data):
    tmpdir, csv_path = dummy_data

    train_tf = get_train_transforms(128)
    dataset = HPADataset(csv_path, tmpdir, transform=train_tf, image_size=128)

    image, target = dataset[0]
    assert image.shape == (4, 128, 128)


def test_val_transforms_deterministic(dummy_data):
    tmpdir, csv_path = dummy_data

    val_tf = get_val_transforms(128)
    dataset = HPADataset(csv_path, tmpdir, transform=val_tf, image_size=128)

    img1, _ = dataset[0]
    img2, _ = dataset[0]
    assert torch.allclose(img1, img2)


def test_label_distribution(dummy_data):
    _, csv_path = dummy_data
    dist = get_label_distribution(csv_path)
    assert len(dist) == NUM_CLASSES
    assert "count" in dist.columns
    assert dist["count"].sum() > 0
