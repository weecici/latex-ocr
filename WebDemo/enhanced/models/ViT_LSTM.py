import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
import timm

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, max_h: int = 200, max_w: int = 200):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4 for 2D PE, got {d_model}")
            
        self.d_model = d_model
        d_half = d_model // 2 # Mỗi chiều H, W chiếm một nửa d_model
        
        pe = torch.zeros(d_model, max_h, max_w)
        
        # div_term cho sin/cos: shape [d_half // 2] -> [128] nếu d_model=512
        div_term = torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))
        
        pos_h = torch.arange(0, max_h).unsqueeze(1).float() # [max_h, 1]
        pos_w = torch.arange(0, max_w).unsqueeze(1).float() # [max_w, 1]
        
        # --- Mã hóa cho chiều Cao (H) ---
        # Tính toán sin/cos: [max_h, d_half // 2]
        sin_h = torch.sin(pos_h * div_term)
        cos_h = torch.cos(pos_h * div_term)
        
        # Chuyển đổi về [d_half // 2, max_h, 1] để broadcast với chiều W
        # Sau đó gán vào pe. PyTorch sẽ tự động broadcast chiều W (max_w)
        pe[0:d_half:2, :, :] = sin_h.transpose(0, 1).unsqueeze(2) 
        pe[1:d_half:2, :, :] = cos_h.transpose(0, 1).unsqueeze(2)
        
        # --- Mã hóa cho chiều Rộng (W) ---
        # Tính toán sin/cos: [max_w, d_half // 2]
        sin_w = torch.sin(pos_w * div_term)
        cos_w = torch.cos(pos_w * div_term)
        
        # Chuyển đổi về [d_half // 2, 1, max_w] để broadcast với chiều H
        pe[d_half::2, :, :] = sin_w.transpose(0, 1).unsqueeze(1)
        pe[d_half+1::2, :, :] = cos_w.transpose(0, 1).unsqueeze(1)
        
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: [1, d_model, max_h, max_w]

    def forward(self, x):
        """
        Args:
            x: Tensor shape [batch_size, d_model, H, W]
        """
        # Cộng PE vào input
        return x + self.pe[:, :, :x.size(2), :x.size(3)]
    

class Encoder(nn.Module):
    def __init__(
        self, 
        enc_out_dim: int, 
        dropout: float
    ) -> None:
        super().__init__()

        self.encoder = timm.create_model(
            'vit_small_patch8_224',
            pretrained=True,
            features_only=True,  
            dynamic_img_size=True,
            drop_rate=dropout
        )
        
        feature_info = self.encoder.feature_info.get_dicts()
        backbone_out_dim = feature_info[-1]['num_chs']

        self.projection = nn.Conv2d(backbone_out_dim, enc_out_dim, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, imgs: Tensor) -> Tensor:
        # features là một list các tensor từ các stages
        features = self.encoder(imgs)                
        encoded_imgs = features[-1]       # Lấy feature map cuối cùng: [B, C_backbone, H', W']
        
        encoded_imgs = self.projection(encoded_imgs) # [B, enc_out_dim, H', W']
        encoded_imgs = self.dropout(encoded_imgs)

        B, C, H, W  = encoded_imgs.shape
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return encoded_imgs  # [B, L, enc_out_dim], L = H*W


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, enc_out_dim, rnn_h_dim, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Các phép chiếu tuyến tính cho Q, K, V
        self.w_q = nn.Linear(rnn_h_dim, d_model, bias=False)
        self.w_k = nn.Linear(enc_out_dim, d_model, bias=False)
        self.w_v = nn.Linear(enc_out_dim, d_model, bias=False)
        
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, projected_key, projected_value):
        # query (từ Decoder): [B, 1, rnn_h_dim]
        # key/value (từ Encoder): [B, L, enc_out_dim]
        batch_size = query.size(0)
        
        # 1. Linear projections và tách đầu (heads)
        # q: [B, n_heads, 1, d_k], k/v: [B, n_heads, L, d_k]
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = projected_key.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = projected_value.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # scores: [B, n_heads, 1, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 3. Kết hợp với Value và gộp heads
        # out: [B, 1, d_model]
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        
        return self.fc_out(out).squeeze(1), attn_weights.squeeze(2)
    

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_emb_dim: int,
        rnn_h_dim: int,
        rnn_o_dim: int,
        enc_out_dim: int,
        att_dim: int,
        n_heads: int,
        dropout: float
    ) -> None:
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, word_emb_dim)
        self.rnn_decoder = nn.LSTMCell(rnn_o_dim + word_emb_dim, rnn_h_dim)
        self.dropout = nn.Dropout(dropout)

        self.mh_cross_attn = MultiHeadCrossAttention(enc_out_dim, rnn_h_dim, d_model=att_dim, n_heads=n_heads, dropout=dropout)

        # Output layers
        self.W_out_context = nn.Linear(rnn_h_dim + att_dim, rnn_o_dim, bias=False)
        self.W_out = nn.Linear(rnn_o_dim, vocab_size, bias=False)
        
        # Init layers
        self.init_wh = nn.Linear(enc_out_dim, rnn_h_dim)
        self.init_wc = nn.Linear(enc_out_dim, rnn_h_dim)
        self.init_wo = nn.Linear(enc_out_dim, rnn_o_dim)

    def init_decoder_states(self, enc_out: Tensor) -> tuple[tuple[Tensor, Tensor], Tensor]:
        """
        Args:
          enc_out: the output of encoder [B, H*W, C]
        Returns:
          h_0, c_0:  h_0 and c_0's shape: [B, rnn_h_dim]
          init_o : the average of enc_out  [B, rnn_o_dim]
          for decoder
        """
        mean_enc_out = enc_out.mean(dim=1)  # [B, H*W, C] -> [B, C]
        h = torch.tanh(self.init_wh(mean_enc_out))   
        c = torch.tanh(self.init_wc(mean_enc_out))
        init_o = torch.tanh(self.init_wo(mean_enc_out))
        return (h, c), init_o

    def forward_step(
        self,
        dec_states: tuple[Tensor, Tensor],
        o_t: Tensor,
        projected_key: Tensor,
        projected_value: Tensor,
        tgt: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor]:
        """Runing one step decoding"""
        # RNN stage
        prev_y = self.embedding(tgt).squeeze(1)  # prev_y's shape: [B, emb_size]
        prev_y = self.dropout(prev_y)
        inp = torch.cat([prev_y, o_t], dim=1)  # inp's shape: [B, emb_size + rnn_o_dim]
        h_t, c_t = self.rnn_decoder(inp, dec_states)  # h_t and c_t's shape: [B, rnn_h_dim]
        h_t = self.dropout(h_t)

        # Attention stage
        # Query lấy từ trạng thái ẩn hiện tại h_t
        query = h_t.unsqueeze(1)            # querry: [B, 1, rnn_h_dim]

        # context_t: [B, att_dim], attn_weights: [B, n_heads, L]
        context_t, attn_weights = self.mh_cross_attn(query, projected_key, projected_value)

        o_t = torch.tanh(self.W_out_context(torch.cat([h_t, context_t], dim=1)))
        o_t = self.dropout(o_t)
        logit = self.W_out(o_t) # [B, vocab_size]

        return (h_t, c_t), o_t, logit, attn_weights
    

class ViT_LSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_emb_dim: int,
        rnn_h_dim: int,
        rnn_o_dim: int,
        enc_out_dim: int,
        att_dim: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(enc_out_dim, dropout)
        self.decoder = Decoder(vocab_size, word_emb_dim, rnn_h_dim, rnn_o_dim, enc_out_dim, att_dim, n_heads, dropout)
        self.uniform = Uniform(0, 1) # Schedule sampling

    
    def forward(self, imgs: Tensor, formulas: Tensor, epsilon: float) -> Tensor:
        """
        Args:
            imgs: [B, 1, H, W]
            formulas: [B, MAX_LEN], each token is a number
            epsilon: probability of the current time step to
                    use the true previous token
        Returns:
            logits: [B, MAX_LEN, VOCAB_SIZE]
        """
        # 1. Encoding
        enc_out = self.encoder(imgs)  # [B, 1, H, W] -> [B, H'*W', 512]

        # 2. Initialize decoder states
        dec_states, o_t = self.decoder.init_decoder_states(enc_out)

        # 3. Loop to decode
        max_len = formulas.size(1)
        logits = []
        prev_logit = None
        projected_key = self.decoder.mh_cross_attn.w_k(enc_out)
        projected_value = self.decoder.mh_cross_attn.w_v(enc_out)
        for t in range(max_len):
            tgt = formulas[:, t : t + 1]  # True label, shape: [B, 1]

            # Schedule sampling
            if prev_logit is not None and self.uniform.sample().item() > epsilon:
                tgt = torch.argmax(prev_logit.detach(), dim=1, keepdim=True) 

            # One step decoding
            dec_states, o_t, logit, attn_weights = self.decoder.forward_step(
                    dec_states, o_t, projected_key, projected_value, tgt
            )  # At each step returns 1 batch, logit: [B, vocab_size]
            
            prev_logit = logit.detach()
            logits.append(logit)

        logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, vocab_size]
        return logits, attn_weights

