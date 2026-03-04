import torch

class BaseConfig:
    # Danh sách các model hỗ trợ
    MODELS = {
        "CNN & Transformer": "weights/CNN_Transformer.pt",
        "ViT & Transformer": "weights/ViT_Transformer.pt"
    }
    
    # Cấu hình mặc định
    DEFAULT_CONFIG = {
        "max_len": 150,
        "beam_size": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_cuda": torch.cuda.is_available(),
        "vocab_path": "weights/vocab.pkl"
    }

class ModelConfig:
    CONFIG = {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 8,
        "d_ff": 1024,
        "max_len": 151
    }

# Biến global 
current_config = BaseConfig.DEFAULT_CONFIG.copy()