"""
Enhanced models: CNN+LSTM and ViT+LSTM with multi-head cross-attention.
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# 2-D Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, max_h: int = 200, max_w: int = 200):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be divisible by 4 for 2D PE, got {d_model}")

        self.d_model = d_model
        d_half = d_model // 2

        pe = torch.zeros(d_model, max_h, max_w)
        div_term = torch.exp(
            torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half)
        )

        pos_h = torch.arange(0, max_h).unsqueeze(1).float()
        pos_w = torch.arange(0, max_w).unsqueeze(1).float()

        sin_h = torch.sin(pos_h * div_term)
        cos_h = torch.cos(pos_h * div_term)
        pe[0:d_half:2, :, :] = sin_h.transpose(0, 1).unsqueeze(2)
        pe[1:d_half:2, :, :] = cos_h.transpose(0, 1).unsqueeze(2)

        sin_w = torch.sin(pos_w * div_term)
        cos_w = torch.cos(pos_w * div_term)
        pe[d_half::2, :, :] = sin_w.transpose(0, 1).unsqueeze(1)
        pe[d_half + 1 :: 2, :, :] = cos_w.transpose(0, 1).unsqueeze(1)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :, : x.size(2), : x.size(3)]


# ---------------------------------------------------------------------------
# CNN Encoder (enhanced, with 2-D PE)
# ---------------------------------------------------------------------------
class _CNNEncoder(nn.Module):
    def __init__(self, enc_out_dim: int, dropout: float) -> None:
        super().__init__()

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, enc_out_dim, 3, 1, 1),
            nn.BatchNorm2d(enc_out_dim),
            nn.ReLU(True),
        )

        self.pos_encoder_2d = PositionalEncoding2D(enc_out_dim, max_h=200, max_w=200)
        self.dropout = nn.Dropout(dropout)

    def forward(self, imgs: Tensor) -> Tensor:
        encoded_imgs = self.cnn_encoder(imgs)
        encoded_imgs = self.pos_encoder_2d(encoded_imgs)
        encoded_imgs = self.dropout(encoded_imgs)

        B, C, H, W = encoded_imgs.shape
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return encoded_imgs


# ---------------------------------------------------------------------------
# ViT Encoder
# ---------------------------------------------------------------------------
class _ViTEncoder(nn.Module):
    def __init__(self, enc_out_dim: int, dropout: float) -> None:
        super().__init__()

        self.encoder = timm.create_model(
            "vit_small_patch8_224",
            pretrained=True,
            features_only=True,
            dynamic_img_size=True,
            drop_rate=dropout,
        )

        feature_info = self.encoder.feature_info.get_dicts()
        backbone_out_dim = feature_info[-1]["num_chs"]

        self.projection = nn.Conv2d(backbone_out_dim, enc_out_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, imgs: Tensor) -> Tensor:
        features = self.encoder(imgs)
        encoded_imgs = features[-1]

        encoded_imgs = self.projection(encoded_imgs)
        encoded_imgs = self.dropout(encoded_imgs)

        B, C, H, W = encoded_imgs.shape
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        return encoded_imgs


# ---------------------------------------------------------------------------
# Multi-Head Cross-Attention
# ---------------------------------------------------------------------------
class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        enc_out_dim: int,
        rnn_h_dim: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(rnn_h_dim, d_model, bias=False)
        self.w_k = nn.Linear(enc_out_dim, d_model, bias=False)
        self.w_v = nn.Linear(enc_out_dim, d_model, bias=False)

        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, projected_key, projected_value):
        batch_size = query.size(0)

        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = projected_key.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = projected_value.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)

        return self.fc_out(out).squeeze(1), attn_weights.squeeze(2)


# ---------------------------------------------------------------------------
# Enhanced Decoder (multi-head cross-attention)
# ---------------------------------------------------------------------------
class _EnhancedDecoder(nn.Module):
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

        self.embedding = nn.Embedding(vocab_size, word_emb_dim)
        self.rnn_decoder = nn.LSTMCell(rnn_o_dim + word_emb_dim, rnn_h_dim)
        self.dropout = nn.Dropout(dropout)

        self.mh_cross_attn = MultiHeadCrossAttention(
            enc_out_dim, rnn_h_dim, d_model=att_dim, n_heads=n_heads, dropout=dropout
        )

        self.W_out_context = nn.Linear(rnn_h_dim + att_dim, rnn_o_dim, bias=False)
        self.W_out = nn.Linear(rnn_o_dim, vocab_size, bias=False)

        self.init_wh = nn.Linear(enc_out_dim, rnn_h_dim)
        self.init_wc = nn.Linear(enc_out_dim, rnn_h_dim)
        self.init_wo = nn.Linear(enc_out_dim, rnn_o_dim)

    def init_decoder_states(
        self, enc_out: Tensor
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        mean_enc_out = enc_out.mean(dim=1)
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
    ) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor, Tensor]:
        prev_y = self.embedding(tgt).squeeze(1)
        prev_y = self.dropout(prev_y)
        inp = torch.cat([prev_y, o_t], dim=1)
        h_t, c_t = self.rnn_decoder(inp, dec_states)
        h_t = self.dropout(h_t)

        query = h_t.unsqueeze(1)
        context_t, attn_weights = self.mh_cross_attn(
            query, projected_key, projected_value
        )

        o_t = torch.tanh(self.W_out_context(torch.cat([h_t, context_t], dim=1)))
        o_t = self.dropout(o_t)
        logit = self.W_out(o_t)

        return (h_t, c_t), o_t, logit, attn_weights


# ---------------------------------------------------------------------------
# Public model classes
# ---------------------------------------------------------------------------
class CNN_LSTM(nn.Module):
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
        self.encoder = _CNNEncoder(enc_out_dim, dropout)
        self.decoder = _EnhancedDecoder(
            vocab_size,
            word_emb_dim,
            rnn_h_dim,
            rnn_o_dim,
            enc_out_dim,
            att_dim,
            n_heads,
            dropout,
        )
        self.uniform = Uniform(0, 1)

    def forward(self, imgs: Tensor, formulas: Tensor, epsilon: float) -> Tensor:
        enc_out = self.encoder(imgs)
        dec_states, o_t = self.decoder.init_decoder_states(enc_out)

        max_len = formulas.size(1)
        logits = []
        prev_logit = None
        projected_key = self.decoder.mh_cross_attn.w_k(enc_out)
        projected_value = self.decoder.mh_cross_attn.w_v(enc_out)
        for t in range(max_len):
            tgt = formulas[:, t : t + 1]
            if prev_logit is not None and self.uniform.sample().item() > epsilon:
                tgt = torch.argmax(prev_logit.detach(), dim=1, keepdim=True)
            dec_states, o_t, logit, attn_weights = self.decoder.forward_step(
                dec_states, o_t, projected_key, projected_value, tgt
            )
            prev_logit = logit.detach()
            logits.append(logit)

        logits = torch.stack(logits, dim=1)
        return logits, attn_weights


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
        self.encoder = _ViTEncoder(enc_out_dim, dropout)
        self.decoder = _EnhancedDecoder(
            vocab_size,
            word_emb_dim,
            rnn_h_dim,
            rnn_o_dim,
            enc_out_dim,
            att_dim,
            n_heads,
            dropout,
        )
        self.uniform = Uniform(0, 1)

    def forward(self, imgs: Tensor, formulas: Tensor, epsilon: float) -> Tensor:
        enc_out = self.encoder(imgs)
        dec_states, o_t = self.decoder.init_decoder_states(enc_out)

        max_len = formulas.size(1)
        logits = []
        prev_logit = None
        projected_key = self.decoder.mh_cross_attn.w_k(enc_out)
        projected_value = self.decoder.mh_cross_attn.w_v(enc_out)
        for t in range(max_len):
            tgt = formulas[:, t : t + 1]
            if prev_logit is not None and self.uniform.sample().item() > epsilon:
                tgt = torch.argmax(prev_logit.detach(), dim=1, keepdim=True)
            dec_states, o_t, logit, attn_weights = self.decoder.forward_step(
                dec_states, o_t, projected_key, projected_value, tgt
            )
            prev_logit = logit.detach()
            logits.append(logit)

        logits = torch.stack(logits, dim=1)
        return logits, attn_weights
