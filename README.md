# Hybrid CNN-Transformer for Protein Subcellular Localization

Multi-label classification of protein subcellular localization from 4-channel fluorescence microscopy images (Human Protein Atlas dataset). Compares three architectures: EfficientNet-B0 baseline, CNN+CBAM, and a hybrid CNN-Transformer model.

## Problem

Given 4-channel fluorescence microscopy images (protein, nucleus, microtubule, endoplasmic reticulum stains), predict which of 28 subcellular organelle classes a protein localizes to. This is a multi-label problem — a single protein can be present in multiple compartments simultaneously.

## Dataset

- **HPA Kaggle Competition** (~31k images, 28 organelle classes): [link](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/data)
- Images are 512×512, 4 channels (RGBY → protein/nucleus/microtubule/ER)

## Setup

```bash
pip install -r requirements.txt
python src/data/download.py
python scripts/train.py --model baseline --epochs 30
```

## References

1. M. Tan & Q. Le — "EfficientNet: Rethinking Model Scaling for CNNs" — ICML, 2019
2. Woo et al. — "CBAM: Convolutional Block Attention Module" — ECCV, 2018
3. Dosovitskiy et al. — "An Image is Worth 16x16 Words: Transformers for Image Recognition" — ICLR, 2021

