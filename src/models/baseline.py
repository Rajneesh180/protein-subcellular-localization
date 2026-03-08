"""EfficientNet-B0 baseline for multi-label classification."""

import torch
import torch.nn as nn
import timm


class EfficientNetBaseline(nn.Module):
    """EfficientNet-B0 with 4-channel input and 28-class sigmoid head."""

    def __init__(self, num_classes: int = 28, pretrained: bool = True, drop_rate: float = 0.3):
        super().__init__()

        # load pretrained efficientnet, modify first conv for 4 channels
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=4,
            num_classes=0,  # remove classifier
            drop_rate=drop_rate,
        )
        self.feature_dim = self.backbone.num_features  # 1280 for b0

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        logits = self.head(features)
        return logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_blocks: int = 2):
        """Unfreeze the last `num_blocks` blocks of EfficientNet."""
        # first unfreeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # efficientnet blocks are in backbone.blocks
        blocks = list(self.backbone.blocks)
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # always unfreeze batch norm in the final conv
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature maps before pooling — used by hybrid model."""
        return self.backbone.forward_features(x)
