# Hybrid CNN-Transformer for Protein Subcellular Localization

Multi-label classification of protein subcellular localization from 4-channel fluorescence microscopy images (Human Protein Atlas dataset). Compares three architectures: EfficientNet-B0 baseline, CNN+CBAM, and a hybrid CNN-Transformer model.

## Problem

Given 4-channel fluorescence microscopy images (protein, nucleus, microtubule, endoplasmic reticulum stains), predict which of 28 subcellular organelle classes a protein localizes to. This is a multi-label problem — a single protein can be present in multiple compartments simultaneously.

## Architecture

Three configurations are trained and compared:

1. **EfficientNet-B0 (baseline)** — pretrained CNN with a 28-neuron sigmoid head
2. **EfficientNet-B0 + CBAM** — channel and spatial attention modules after the CNN backbone
3. **EfficientNet-B0 + Transformer** — CNN feature maps are patchified and passed through a cross-attention Transformer block before classification

All models share the same training pipeline, augmentation, and evaluation protocol for a fair comparison.

## Dataset

- **HPA Kaggle Competition** (~31k images, 28 organelle classes): [link](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/data)
- **HPA Full Database** (supplementary): [proteinatlas.org](https://www.proteinatlas.org/humanproteome/subcellular)
- Images are 512×512, 4 channels (RGBY → protein/nucleus/microtubule/ER)

## Project Structure

```
├── configs/
│   └── config.yaml            # training hyperparameters
├── src/
│   ├── data/
│   │   ├── dataset.py         # HPA dataset class
│   │   ├── augmentation.py    # augmentation pipeline
│   │   └── download.py        # kaggle data download helper
│   ├── models/
│   │   ├── baseline.py        # EfficientNet-B0 baseline
│   │   ├── cbam.py            # CBAM attention module
│   │   ├── hybrid.py          # CNN + Transformer hybrid
│   │   └── factory.py         # model factory
│   ├── training/
│   │   ├── trainer.py         # training loop
│   │   └── losses.py          # loss functions
│   ├── evaluation/
│   │   ├── metrics.py         # F1, ROC-AUC computation
│   │   └── gradcam.py         # Grad-CAM + attention viz
│   └── serving/
│       ├── app.py             # FastAPI inference endpoint
│       └── schemas.py         # request/response schemas
├── scripts/
│   ├── train.py               # training entry point
│   └── evaluate.py            # evaluation entry point
├── tests/
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_api.py
├── notebooks/
│   └── eda.ipynb              # exploratory data analysis
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml
```

## Setup

```bash
# clone
git clone https://github.com/Rajneesh180/protein-subcellular-localization.git
cd protein-subcellular-localization

# install deps
pip install -r requirements.txt

# download HPA data (requires kaggle API key)
python src/data/download.py

# train baseline
python scripts/train.py --model baseline --epochs 30

# train hybrid
python scripts/train.py --model hybrid --epochs 30

# evaluate all models
python scripts/evaluate.py --checkpoint checkpoints/
```

## Training

All training runs are tracked with MLflow. Default config is in `configs/config.yaml`.

```bash
# start mlflow ui
mlflow ui --port 5000

# train with custom config
python scripts/train.py --model hybrid --lr 1e-4 --batch-size 16
```

## Inference API

```bash
# run locally
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# or via docker
docker-compose up
```

POST a 4-channel image to `/predict`:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample_image.png"
```

## Evaluation

| Model | F1-macro | ROC-AUC (macro) |
|-------|----------|-----------------|
| EfficientNet-B0 (baseline) | — | — |
| EfficientNet-B0 + CBAM | — | — |
| EfficientNet-B0 + Transformer | — | — |

Results will be filled after training completes on the full dataset.

## Interpretability

Grad-CAM heatmaps and Transformer attention maps are generated for qualitative analysis. See `scripts/evaluate.py --gradcam` for visualization output.

## References

1. S. Aggarwal et al. — "A CNN-Based Framework for Classification of Protein Localization Using Confocal Microscopy Images" — IEEE Access, 2022
2. S. Aggarwal et al. — "Protein Subcellular Localization Prediction by Concatenation of Convolutional Blocks" — IEEE Access, 2022
3. M. Tan & Q. Le — "EfficientNet: Rethinking Model Scaling for CNNs" — ICML, 2019
4. Woo et al. — "CBAM: Convolutional Block Attention Module" — ECCV, 2018
5. Dosovitskiy et al. — "An Image is Worth 16x16 Words: Transformers for Image Recognition" — ICLR, 2021

## License

MIT

