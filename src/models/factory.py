"""Model factory — instantiate model by name."""

from src.models.baseline import EfficientNetBaseline
from src.models.cbam import EfficientNetCBAM
from src.models.hybrid import HybridCNNTransformer


MODEL_REGISTRY = {
    "baseline": EfficientNetBaseline,
    "cbam": EfficientNetCBAM,
    "hybrid": HybridCNNTransformer,
}


def build_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Options: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
