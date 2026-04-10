# Hybrid CNN-Transformer for Protein Subcellular Localization

Multi-label classification of protein subcellular localization from 4-channel fluorescence microscopy images (Human Protein Atlas). Compares three architectures — EfficientNet-B0, CNN+CBAM, and a hybrid CNN-Transformer — then uses the best model to build a protein mislocalisation detector.

## Problem

Fluorescence microscopy images from the Human Protein Atlas contain four staining channels (microtubule, protein, nucleus, ER). Each protein can localise to multiple subcellular compartments simultaneously, making this a multi-label classification problem across 28 organelle classes. Accurate localization matters for understanding protein function; deviations from expected patterns can signal disease states or drug effects.

## Architecture

| Model | Params | Description |
|-------|--------|-------------|
| EfficientNet-B0 | 4.04M | Pretrained CNN backbone → 28-neuron sigmoid head |
| EfficientNet-B0 + CBAM | 4.25M | Adds channel + spatial attention after the backbone |
| EfficientNet-B0 + Transformer | 5.94M | CNN features → patch embedding → 2-block Transformer → CLS token classifier |

All models share the same 4-channel input adapter, training pipeline (Focal Loss, differential LR, warmup + ReduceLROnPlateau), augmentation strategy, and evaluation protocol.

## Results

Trained on 787 images from [proteinatlas.org](https://www.proteinatlas.org/humanproteome/subcellular) covering 25 of 28 HPA organelle classes. Validation split: 20% stratified.

| Model | F1-macro | F1-micro | ROC-AUC (macro) |
|-------|----------|----------|-----------------|
| EfficientNet-B0 (baseline) | 0.1447 | 0.3077 | 0.6862 |
| EfficientNet-B0 + CBAM | 0.1650 | 0.3418 | 0.7165 |
| **EfficientNet-B0 + Transformer** | **0.2010** | **0.3617** | **0.7392** |

F1-macro is low because of extreme class imbalance across 28 labels with only 787 samples — but the hybrid model consistently outperforms the baselines, and the Transformer attention maps reveal meaningful spatial localization.

## Mislocalisation Detection

The trained hybrid model powers a mislocalisation screening pipeline: given a microscopy image, predicted localization is compared against expected compartments from a reference database (built from training co-occurrence statistics). Deviations are scored as anomalies.

```bash
python scripts/detect_mislocalisation.py --model hybrid --output-dir results/
```

This generates:
- **Co-occurrence matrix**: which compartments tend to appear together in the training data
- **Mislocalisation report**: flagged samples with predicted vs expected labels, anomaly scores, and protein channel images

Use case: screen for drug-induced or disease-related protein trafficking changes without manual annotation.

## Dataset

- 787 images scraped from [proteinatlas.org](https://www.proteinatlas.org/humanproteome/subcellular) (4-channel RGBY PNGs)
- 25 of 28 HPA organelle classes represented
- Images resized to 256×256 for training
- Each image has four channels: red (microtubule), green (protein of interest), blue (nucleus), yellow (ER)

## Project Structure

```
├── configs/
│   └── config.yaml                  # training hyperparameters
├── src/
│   ├── data/
│   │   ├── dataset.py               # HPA dataset class
│   │   ├── augmentation.py          # augmentation pipeline
│   │   └── download.py              # proteinatlas scraper
│   ├── models/
│   │   ├── baseline.py              # EfficientNet-B0
│   │   ├── cbam.py                  # CBAM attention module
│   │   ├── hybrid.py                # CNN + Transformer hybrid
│   │   └── factory.py               # model registry
│   ├── training/
│   │   ├── trainer.py               # training loop w/ MLflow
│   │   └── losses.py                # Focal loss + pos weights
│   ├── evaluation/
│   │   ├── metrics.py               # F1, ROC-AUC computation
│   │   └── gradcam.py               # Grad-CAM + attention viz
│   ├── analysis/
│   │   └── mislocalisation.py       # mislocalisation detector
│   └── serving/
│       ├── app.py                   # FastAPI endpoint
│       └── schemas.py               # request/response schemas
├── scripts/
│   ├── train.py                     # training CLI
│   ├── evaluate.py                  # evaluation CLI
│   └── detect_mislocalisation.py    # mislocalisation CLI
├── tests/
├── notebooks/
│   └── results.ipynb                # training results + analysis
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/Rajneesh180/protein-subcellular-localization.git
cd protein-subcellular-localization
pip install -r requirements.txt
```

## Training

```bash
# train all three models
python scripts/train.py --model baseline --epochs 15
python scripts/train.py --model cbam --epochs 15
python scripts/train.py --model hybrid --epochs 15

# evaluate
python scripts/evaluate.py --model hybrid --gradcam

# MLflow dashboard
mlflow ui --port 5000
```

Training uses Focal Loss (α=0.75, γ=2.0), differential learning rates (backbone 10× lower), warmup for 2 epochs, and early stopping with patience 7.

## Inference API

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
# or
docker-compose up
```

```bash
curl -X POST http://localhost:8000/predict -F "file=@sample_image.png"
```

## Interpretability

Grad-CAM heatmaps show which image regions drive each prediction. The hybrid model's Transformer attention maps reveal how the CLS token aggregates spatial information across patches — both layers learn distinct localization patterns.

See `notebooks/results.ipynb` for attention map visualizations and per-class analysis.

## References

1. S. Aggarwal et al. — "A CNN-Based Framework for Classification of Protein Localization Using Confocal Microscopy Images" — IEEE Access, 2022
2. S. Aggarwal et al. — "Protein Subcellular Localization Prediction by Concatenation of Convolutional Blocks" — IEEE Access, 2022
3. M. Tan & Q. Le — "EfficientNet: Rethinking Model Scaling for CNNs" — ICML, 2019
4. Woo et al. — "CBAM: Convolutional Block Attention Module" — ECCV, 2018
5. Dosovitskiy et al. — "An Image is Worth 16x16 Words: Transformers for Image Recognition" — ICLR, 2021

## License

MIT

