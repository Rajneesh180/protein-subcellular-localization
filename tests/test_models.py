"""Tests for model architectures."""

import pytest
import torch

from src.models.baseline import EfficientNetBaseline
from src.models.cbam import EfficientNetCBAM, CBAM, ChannelAttention, SpatialAttention
from src.models.hybrid import HybridCNNTransformer, TransformerBlock
from src.models.factory import build_model


@pytest.fixture
def dummy_input():
    return torch.randn(2, 4, 128, 128)


class TestBaseline:
    def test_forward_shape(self, dummy_input):
        model = EfficientNetBaseline(num_classes=28, pretrained=False)
        out = model(dummy_input)
        assert out.shape == (2, 28)

    def test_freeze_unfreeze(self):
        model = EfficientNetBaseline(pretrained=False)
        model.freeze_backbone()
        frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        assert frozen > 0

        model.unfreeze_backbone(num_blocks=1)
        unfrozen = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        assert unfrozen > 0

    def test_feature_maps(self, dummy_input):
        model = EfficientNetBaseline(pretrained=False)
        fmaps = model.get_feature_maps(dummy_input)
        assert fmaps.dim() == 4
        assert fmaps.shape[1] == 1280  # efficientnet_b0 features


class TestCBAM:
    def test_channel_attention(self):
        ca = ChannelAttention(64, reduction=16)
        x = torch.randn(2, 64, 8, 8)
        out = ca(x)
        assert out.shape == x.shape

    def test_spatial_attention(self):
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 64, 8, 8)
        out = sa(x)
        assert out.shape == x.shape

    def test_cbam_module(self):
        cbam = CBAM(128, reduction=16, kernel_size=7)
        x = torch.randn(2, 128, 16, 16)
        out = cbam(x)
        assert out.shape == x.shape

    def test_efficientnet_cbam_forward(self, dummy_input):
        model = EfficientNetCBAM(num_classes=28, pretrained=False)
        out = model(dummy_input)
        assert out.shape == (2, 28)


class TestHybrid:
    def test_transformer_block(self):
        block = TransformerBlock(dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_hybrid_forward(self, dummy_input):
        model = HybridCNNTransformer(
            num_classes=28, pretrained=False,
            embed_dim=64, num_heads=4, depth=1,
        )
        out = model(dummy_input)
        assert out.shape == (2, 28)

    def test_attention_maps(self, dummy_input):
        model = HybridCNNTransformer(
            num_classes=28, pretrained=False,
            embed_dim=64, num_heads=4, depth=2,
        )
        attn_maps = model.get_attention_maps(dummy_input)
        assert len(attn_maps) == 2  # depth=2


class TestFactory:
    def test_build_baseline(self):
        model = build_model("baseline", num_classes=28, pretrained=False)
        assert isinstance(model, EfficientNetBaseline)

    def test_build_cbam(self):
        model = build_model("cbam", num_classes=28, pretrained=False)
        assert isinstance(model, EfficientNetCBAM)

    def test_build_hybrid(self):
        model = build_model("hybrid", num_classes=28, pretrained=False, embed_dim=64, num_heads=4, depth=1)
        assert isinstance(model, HybridCNNTransformer)

    def test_unknown_model(self):
        with pytest.raises(ValueError):
            build_model("unknown_model")
