"""CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)."""

import torch
import torch.nn as nn
import timm


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attn = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(combined))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class EfficientNetCBAM(nn.Module):
    """EfficientNet-B0 + CBAM attention on top of feature maps."""

    def __init__(
        self,
        num_classes: int = 28,
        pretrained: bool = True,
        drop_rate: float = 0.3,
        reduction: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=4,
            num_classes=0,
            drop_rate=drop_rate,
        )
        feat_dim = self.backbone.num_features  # 1280

        self.cbam = CBAM(feat_dim, reduction=reduction, kernel_size=kernel_size)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        attended = self.cbam(features)
        logits = self.head(attended)
        return logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_blocks: int = 2):
        for param in self.backbone.parameters():
            param.requires_grad = False
        blocks = list(self.backbone.blocks)
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return CBAM-attended feature maps before pooling — used for Grad-CAM."""
        features = self.backbone.forward_features(x)
        return self.cbam(features)
