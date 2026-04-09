"""Hybrid CNN-Transformer model: EfficientNet backbone + cross-attention head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PatchEmbedFromFeatures(nn.Module):
    """Convert CNN feature maps to a sequence of patch embeddings for the Transformer.

    Takes (B, C, H, W) feature maps → (B, H*W, embed_dim).
    """

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                  # (B, embed_dim, H, W)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        return x


class HybridCNNTransformer(nn.Module):
    """EfficientNet-B0 CNN backbone + lightweight Transformer attention head.

    CNN extracts spatial features → patchified → Transformer encodes global
    dependencies → classification head.
    """

    def __init__(
        self,
        num_classes: int = 28,
        pretrained: bool = True,
        drop_rate: float = 0.3,
        embed_dim: int = 256,
        num_heads: int = 4,
        depth: int = 2,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        # CNN backbone
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=4,
            num_classes=0,
            drop_rate=drop_rate,
        )
        cnn_dim = self.backbone.num_features  # 1280

        # project CNN features to Transformer dim
        self.patch_embed = PatchEmbedFromFeatures(cnn_dim, embed_dim)

        # learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # positional embedding — will be interpolated if spatial size changes
        # for efficientnet_b0 with 256 input, feature map is 8x8 = 64 patches + 1 cls
        self.pos_embed = nn.Parameter(torch.zeros(1, 65, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # classification head
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_classes),
        )

    def _interpolate_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Adjust positional embedding if sequence length doesn't match via 2D interpolation."""
        n = x.shape[1]
        if n == self.pos_embed.shape[1]:
            return self.pos_embed

        cls_embed = self.pos_embed[:, :1, :]
        patch_embed = self.pos_embed[:, 1:, :]
        old_n = patch_embed.shape[1]
        old_h = int(old_n ** 0.5)
        new_n = n - 1
        new_h = int(new_n ** 0.5)

        # reshape to 2D grid, interpolate, flatten back
        patch_embed = patch_embed.reshape(1, old_h, old_h, -1).permute(0, 3, 1, 2)
        patch_embed = F.interpolate(patch_embed, size=(new_h, new_h), mode="bilinear", align_corners=False)
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_h * new_h, -1)
        return torch.cat([cls_embed, patch_embed], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # extract CNN features
        features = self.backbone.forward_features(x)  # (B, 1280, H', W')

        # convert to patch sequence
        tokens = self.patch_embed(features)  # (B, N, embed_dim)

        # prepend CLS token
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, embed_dim)

        # add positional embedding
        tokens = tokens + self._interpolate_pos_embed(tokens)

        # Transformer encoding
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        # classify from CLS token
        cls_out = tokens[:, 0]
        logits = self.head(cls_out)
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

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """Extract attention weights from each Transformer block for visualization."""
        features = self.backbone.forward_features(x)
        tokens = self.patch_embed(features)
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self._interpolate_pos_embed(tokens)

        attn_maps = []
        for block in self.transformer:
            # manually run the block from forward, capturing attention weights
            x_norm = block.norm1(tokens)
            attn_out, attn_weights = block.attn(x_norm, x_norm, x_norm, need_weights=True)
            tokens = tokens + attn_out
            tokens = tokens + block.mlp(block.norm2(tokens))
            attn_maps.append(attn_weights.detach())

        return attn_maps
