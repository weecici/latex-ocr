import torch

class BaseConfig:
    # Danh sách các model hỗ trợ
    MODELS = {
        "CNN & LSTM": "weights/CNN_LSTM.pt",
        "ResNet18 & LSTM": "weights/ResNet18_LSTM.pt"
    }
    
    # Cấu hình mặc định
    DEFAULT_CONFIG = {
        "max_len": 150,
        "beam_size": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_cuda": True,
        "vocab_path": "weights/vocab.pkl"
    }

class ModelConfig:
    CONFIG = {
        "word_emb_dim": 80,
        "rnn_h_dim": 512, 
        "rnn_o_dim": 512,
        "enc_out_dim": 512,
        "attn_dim": 512,
        "dropout": 0.2
    }

# Biến global để app có thể sửa đổi trong runtime nếu cần
current_config = BaseConfig.DEFAULT_CONFIG.copy()