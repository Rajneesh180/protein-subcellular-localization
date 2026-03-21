"""Grad-CAM and Transformer attention visualization."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data.dataset import HPA_LABELS


def generate_gradcam(model, images, target_layer, target_class=None, device="cuda"):
    """Generate Grad-CAM heatmaps for a batch of images.

    Args:
        model: trained model
        images: (B, 4, H, W) tensor
        target_layer: layer to extract gradients from
        target_class: if None, uses predicted class
        device: computation device

    Returns:
        list of (H, W) numpy heatmaps
    """
    model.eval()
    model.to(device)
    images = images.to(device)

    cam = GradCAM(model=model, target_layers=[target_layer])

    if target_class is not None:
        # for multi-label, we define custom targets
        from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
        targets = [BinaryClassifierOutputTarget(target_class)] * images.shape[0]
    else:
        targets = None

    grayscale_cams = cam(input_tensor=images, targets=targets)
    return grayscale_cams


def visualize_gradcam_grid(
    images: torch.Tensor,
    heatmaps: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str = "gradcam_grid.png",
    num_samples: int = 8,
):
    """Plot a grid of images with Grad-CAM overlays."""
    n = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))

    for i in range(n):
        # show protein channel (green) as base image
        img_np = images[i, 1].cpu().numpy()  # green = protein channel
        img_rgb = np.stack([img_np] * 3, axis=-1)
        img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)

        # original image
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title("Protein channel")
        axes[i, 0].axis("off")

        # heatmap overlay
        cam_image = show_cam_on_image(img_rgb.astype(np.float32), heatmaps[i], use_rgb=True)
        axes[i, 1].imshow(cam_image)
        axes[i, 1].set_title("Grad-CAM")
        axes[i, 1].axis("off")

        # predicted labels
        pred_labels = np.where(predictions[i] >= 0.5)[0]
        true_labels = np.where(targets[i] == 1)[0]
        pred_text = ", ".join([HPA_LABELS[j] for j in pred_labels]) or "none"
        true_text = ", ".join([HPA_LABELS[j] for j in true_labels]) or "none"

        axes[i, 2].text(0.1, 0.6, f"Pred: {pred_text}", fontsize=9, wrap=True)
        axes[i, 2].text(0.1, 0.3, f"True: {true_text}", fontsize=9, color="green", wrap=True)
        axes[i, 2].axis("off")
        axes[i, 2].set_title("Labels")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM grid to {save_path}")


def visualize_attention_maps(
    model,
    images: torch.Tensor,
    save_path: str = "attention_maps.png",
    device: str = "cuda",
):
    """Visualize Transformer self-attention maps (only for hybrid model)."""
    if not hasattr(model, "get_attention_maps"):
        print("Model does not have attention maps — skipping")
        return

    model.eval()
    model.to(device)
    images = images.to(device)

    attn_maps = model.get_attention_maps(images[:4])  # first 4 samples

    num_layers = len(attn_maps)
    fig, axes = plt.subplots(4, num_layers + 1, figsize=(5 * (num_layers + 1), 20))

    for i in range(4):
        # protein channel
        img_np = images[i, 1].cpu().numpy()
        axes[i, 0].imshow(img_np, cmap="gray")
        axes[i, 0].set_title(f"Sample {i}")
        axes[i, 0].axis("off")

        for layer_idx, attn in enumerate(attn_maps):
            # average over heads, show CLS token attention to patches
            cls_attn = attn[i].mean(dim=0)[0, 1:]  # skip CLS-to-CLS
            n_patches = cls_attn.shape[0]
            h = w = int(n_patches ** 0.5)
            if h * w != n_patches:
                h = w = int(np.ceil(n_patches ** 0.5))
                cls_attn = F.pad(cls_attn, (0, h * w - n_patches))

            attn_map = cls_attn.reshape(h, w).cpu().numpy()
            axes[i, layer_idx + 1].imshow(attn_map, cmap="hot")
            axes[i, layer_idx + 1].set_title(f"Layer {layer_idx} attention")
            axes[i, layer_idx + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention maps to {save_path}")
