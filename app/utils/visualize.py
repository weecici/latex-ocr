"""
Unified attention visualization for both baseline (single-head) and
enhanced (multi-head) models.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch


def visualize_attention(image, tokens, alphas, H_prime, W_prime, skip_special=True):
    """
    Visualize attention maps overlaid on the input image.

    Parameters
    ----------
    image : numpy array or PIL Image
        The original input image.
    tokens : list[str]
        Decoded token strings.
    alphas : torch.Tensor or np.ndarray
        Attention weights. Shape can be:
          - [seq_len, L]             (baseline, single-head)
          - [seq_len, n_heads, L]    (enhanced, multi-head)
    H_prime, W_prime : int
        Spatial dimensions of the encoder feature map.
    skip_special : bool
        Whether to skip special tokens (<pad>, <eos>, etc.).
    """
    if isinstance(alphas, torch.Tensor):
        alphas = alphas.detach().cpu().numpy()

    # Average across heads if multi-head
    if len(alphas.shape) == 3:
        alphas = alphas.mean(axis=1)

    # Filter special tokens
    processed_tokens = []
    processed_alphas = []
    special_tokens = {"<pad>", "<eos>", "<sos>", "<s>", "</s>", "PAD", "EOS"}

    for i, token in enumerate(tokens):
        if skip_special and token in special_tokens:
            continue
        if i < len(alphas):
            processed_tokens.append(token)
            processed_alphas.append(alphas[i])

    num_tokens = len(processed_tokens)
    if num_tokens == 0:
        return None

    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]

    n_cols = 4
    n_rows = (num_tokens + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten() if num_tokens > 1 else [axes]

    for i in range(num_tokens):
        att_map = processed_alphas[i].reshape(H_prime, W_prime)

        # Min-max normalization
        att_min, att_max = att_map.min(), att_map.max()
        if att_max - att_min > 1e-8:
            att_map = (att_map - att_min) / (att_max - att_min)

        att_map_resized = cv2.resize(att_map, (img_w, img_h))

        axes[i].imshow(img_array)
        axes[i].imshow(att_map_resized, cmap="jet", alpha=0.5)
        axes[i].set_title(f"Token: {processed_tokens[i]}", fontsize=10)
        axes[i].axis("off")

    for j in range(num_tokens, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig
