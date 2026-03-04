import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def visualize_attention(image, tokens, alphas, H_prime, W_prime):
    """
    image: PIL Image hoặc numpy array (Ảnh gốc)
    tokens: List[str] (Danh sách các ký tự đã split)
    alphas: torch.Tensor hoặc np.array shape [Seq_Len, H_prime * W_prime]
    H_prime, W_prime: Kích thước feature map (ví dụ: H/8, W/8)
    """
    # Chuyển image sang numpy nếu là PIL
    img_array = np.array(image)
    
    # Số lượng token thực tế (không vượt quá số lượng alpha frames có sẵn)
    num_tokens = min(len(tokens), alphas.shape[0])
    
    # Tính toán số hàng/cột cho subplot (Cố định 4 cột mỗi hàng)
    n_cols = 4
    n_rows = (num_tokens + n_cols - 1) // n_cols
    
    # Khởi tạo Figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    for i in range(num_tokens):
        # 1. Lấy alpha của bước t và reshape về [H_prime, W_prime]
        att_map = alphas[i].reshape(H_prime, W_prime)
        if isinstance(att_map, torch.Tensor):
            att_map = att_map.detach().cpu().numpy()

        # 2. Resize attention map bằng kích thước ảnh gốc để overlay
        # Sử dụng nội suy INTER_CUBIC để map mượt mà hơn
        att_map_resized = cv2.resize(att_map, (img_array.shape[1], img_array.shape[0]))

        # 3. Vẽ ảnh gốc
        axes[i].imshow(img_array)
        
        # 4. Vẽ Attention Map đè lên (Sử dụng colormap 'jet' hoặc 'viridis')
        # alpha=0.5 giúp nhìn xuyên qua lớp heatmap thấy ảnh gốc
        im = axes[i].imshow(att_map_resized, cmap='jet', alpha=0.5)
        
        axes[i].set_title(f"Token: {tokens[i]}", fontsize=12)
        axes[i].axis('off')

    # Ẩn các trục thừa nếu số lượng token không lấp đầy grid
    for j in range(num_tokens, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig