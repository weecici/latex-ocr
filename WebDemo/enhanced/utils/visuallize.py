import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

def visualize_attention(image, tokens, alphas, H_prime, W_prime, skip_special=True):
    """
    image: PIL Image hoặc numpy array
    tokens: List[str] (Ví dụ: ['x', '=', 'y', '<eos>'])
    alphas: torch.Tensor shape [Seq_Len, Heads, H_prime * W_prime] 
            hoặc [Seq_Len, H_prime * W_prime]
    H_prime, W_prime: Kích thước feature map
    """
    # 1. Chuyển đổi alphas sang numpy và xử lý Multi-head
    if isinstance(alphas, torch.Tensor):
        alphas = alphas.detach().cpu().numpy()
    
    # Nếu có chiều Heads (3D), tính trung bình các đầu attention
    if len(alphas.shape) == 3:
        alphas = alphas.mean(axis=1) # Shape còn lại: [Seq_Len, H_prime * W_prime]

    # 2. Lọc bỏ các token không cần thiết (PAD, EOS)
    processed_tokens = []
    processed_alphas = []
    special_tokens = ['<pad>', '<eos>', '<sos>', 'PAD', 'EOS']
    
    for i, token in enumerate(tokens):
        if skip_special and (token in special_tokens):
            continue
        processed_tokens.append(token)
        processed_alphas.append(alphas[i])
    
    num_tokens = len(processed_tokens)
    if num_tokens == 0:
        print("Không có token hợp lệ để hiển thị.")
        return None

    # 3. Chuyển image sang numpy
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]

    # 4. Thiết lập Grid hiển thị (Cố định 4 cột)
    n_cols = 4
    n_rows = (num_tokens + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten() if num_tokens > 1 else [axes]

    for i in range(num_tokens):
        # Lấy bản đồ attention và reshape
        att_map = processed_alphas[i].reshape(H_prime, W_prime)
        
        # Chuẩn hóa Min-Max để heatmap nổi bật hơn
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)

        # Resize về kích thước ảnh gốc
        att_map_resized = cv2.resize(att_map, (img_w, img_h))

        # Vẽ ảnh gốc và đè heatmap
        axes[i].imshow(img_array)
        # Sử dụng 'jet' hoặc 'viridis', alpha=0.6 để cân bằng độ rõ
        im = axes[i].imshow(att_map_resized, cmap='jet', alpha=0.5) 
        
        axes[i].set_title(f"Token: {processed_tokens[i]}", fontsize=10)
        axes[i].axis('off')

    # Ẩn các ô trống còn thừa
    for j in range(num_tokens, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig