"""HPA multi-label dataset for 4-channel fluorescence microscopy images."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


# official HPA class names
HPA_LABELS = [
    "Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center",
    "Nuclear speckles", "Nuclear bodies", "Endoplasmic reticulum", "Golgi apparatus",
    "Intermediate filaments", "Actin filaments", "Microtubules", "Mitotic spindle",
    "Centrosome", "Plasma membrane", "Mitochondria", "Aggresome",
    "Cytosol", "Vesicles and puncta", "Negative", "Endosomes",
    "Lysosomes", "Lipid droplets", "Peroxisomes", "Autophagosomes",
    "Focal adhesion sites", "Cell junctions", "Rods and rings", "Undefined",
]

NUM_CLASSES = len(HPA_LABELS)
CHANNELS = ["red", "green", "blue", "yellow"]  # microtubule, protein, nucleus, ER


class HPADataset(Dataset):
    """Load 4-channel HPA images with multi-label targets.

    Each sample has four single-channel PNG files: {id}_{color}.png
    Target is a 28-dim binary vector.
    """

    def __init__(self, csv_path: str, image_dir: str, transform=None, image_size: int = 512):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size

        # parse multi-label targets
        self.targets = self._encode_labels()

    def _encode_labels(self) -> np.ndarray:
        encoded = np.zeros((len(self.df), NUM_CLASSES), dtype=np.float32)
        for i, row in self.df.iterrows():
            labels = str(row["Target"]).split()
            for lbl in labels:
                encoded[i, int(lbl)] = 1.0
        return encoded

    def _load_image(self, image_id: str) -> np.ndarray:
        """Load 4 channels and stack into (H, W, 4) array."""
        channels = []
        for color in CHANNELS:
            path = os.path.join(self.image_dir, f"{image_id}_{color}.png")
            img = Image.open(path).convert("L")
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            channels.append(np.asarray(img, dtype=np.float32))

        # stack: (H, W, 4)
        stacked = np.stack(channels, axis=-1)
        return stacked

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["Id"])

        image = self._load_image(image_id)
        target = self.targets[idx]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # convert to tensor: (4, H, W) normalized to [0, 1]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1) / 255.0

        target = torch.from_numpy(target)

        return image, target


def get_label_distribution(csv_path: str) -> pd.DataFrame:
    """Compute per-class frequency for analysis."""
    df = pd.read_csv(csv_path)
    counts = {i: 0 for i in range(NUM_CLASSES)}
    for targets in df["Target"]:
        for lbl in str(targets).split():
            counts[int(lbl)] += 1

    dist = pd.DataFrame({
        "class_id": list(counts.keys()),
        "class_name": HPA_LABELS,
        "count": list(counts.values()),
    })
    dist["frequency"] = dist["count"] / len(df)
    return dist.sort_values("count", ascending=False).reset_index(drop=True)
