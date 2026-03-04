"""
Transformer-based models: CNN+Transformer and ViT+Transformer with KV-cache.
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# Positional Encodings
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


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


# ---------------------------------------------------------------------------
# CNN Encoder (for CNN+Transformer)
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
# Multi-Head Attention (self + cross, with KV-cache support)
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
        precomputed_kv=None,
        past_kv=None,
        use_cache=False,
    ):
        batch_size = query.size(0)

        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        if precomputed_kv is not None:
            k, v = precomputed_kv
            present_kv = None
        else:
            k = (
                self.w_k(key)
                .view(batch_size, -1, self.n_heads, self.d_k)
                .transpose(1, 2)
            )
            v = (
                self.w_v(value)
                .view(batch_size, -1, self.n_heads, self.d_k)
                .transpose(1, 2)
            )

            if past_kv is not None:
                past_k, past_v = past_kv
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            present_kv = (k, v) if use_cache else None

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.fc_out(out), attn_weights, present_kv

    def precompute_kv(self, key_value):
        batch_size = key_value.size(0)
        k = (
            self.w_k(key_value)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.w_v(key_value)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        return (k, v)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ---------------------------------------------------------------------------
# Transformer Decoder Layer
# ---------------------------------------------------------------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        enc_out,
        tgt_mask=None,
        src_mask=None,
        precomputed_enc_kv=None,
        past_self_kv=None,
        use_cache=False,
    ):
        attn_out, _, present_self_kv = self.self_attn(
            tgt, tgt, tgt, mask=tgt_mask, past_kv=past_self_kv, use_cache=use_cache
        )
        tgt = self.norm1(tgt + self.dropout(attn_out))

        attn_out, cross_attn_weights, _ = self.cross_attn(
            tgt, enc_out, enc_out, mask=src_mask, precomputed_kv=precomputed_enc_kv
        )
        tgt = self.norm2(tgt + self.dropout(attn_out))

        ffn_out = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(ffn_out))

        return tgt, cross_attn_weights, present_self_kv


# ---------------------------------------------------------------------------
# Transformer Decoder
# ---------------------------------------------------------------------------
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int,
        dropout: float,
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding1D(d_model, max_len)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.embedding_scale = math.sqrt(d_model)

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0

    def generate_padding_mask(self, tgt, pad_idx=0):
        mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def combine_masks(self, causal_mask, padding_mask):
        return causal_mask & padding_mask

    def forward(self, tgt, enc_out, tgt_mask=None, precomputed_enc_kv=None, pad_idx=0):
        tgt_emb = self.embedding(tgt) * self.embedding_scale
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        if tgt_mask is None:
            seq_len = tgt.size(1)
            device = tgt.device

            causal_mask = self.generate_causal_mask(seq_len, device)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            padding_mask = self.generate_padding_mask(tgt, pad_idx)

            tgt_mask = self.combine_masks(causal_mask, padding_mask)

        x = tgt_emb
        for i, layer in enumerate(self.layers):
            layer_kv = precomputed_enc_kv[i] if precomputed_enc_kv else None
            x, cross_attn_weights, _ = layer(
                x, enc_out, tgt_mask=tgt_mask, precomputed_enc_kv=layer_kv
            )

        logits = self.fc_out(x)
        return logits, cross_attn_weights

    def precompute_encoder_kv(self, enc_out):
        precomputed_kv = []
        for layer in self.layers:
            kv = layer.cross_attn.precompute_kv(enc_out)
            precomputed_kv.append(kv)
        return precomputed_kv


# ---------------------------------------------------------------------------
# Public model classes
# ---------------------------------------------------------------------------
class CNN_Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int,
        dropout: float,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.encoder = _CNNEncoder(d_model, dropout)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
        )
        self.uniform = Uniform(0, 1)
        self.pad_idx = pad_idx

    def forward(self, imgs: Tensor, formulas: Tensor, epsilon: float = 1.0):
        batch_size = imgs.size(0)
        max_len = formulas.size(1)

        enc_out = self.encoder(imgs)
        precomputed_enc_kv = self.decoder.precompute_encoder_kv(enc_out)

        if self.training and epsilon < 0.9:
            logits_list = []
            prev_tokens = formulas[:, 0:1]

            for t in range(max_len):
                logits, attn_weights = self.decoder(
                    prev_tokens,
                    enc_out,
                    precomputed_enc_kv=precomputed_enc_kv,
                    pad_idx=self.pad_idx,
                )
                current_logit = logits[:, -1:, :]
                logits_list.append(current_logit)

                if t < max_len - 1:
                    if self.uniform.sample().item() < epsilon:
                        next_token = formulas[:, t + 1 : t + 2]
                    else:
                        next_token = torch.argmax(current_logit, dim=-1)
                    prev_tokens = torch.cat([prev_tokens, next_token], dim=1)

            logits = torch.cat(logits_list, dim=1)
        else:
            logits, attn_weights = self.decoder(
                formulas,
                enc_out,
                precomputed_enc_kv=precomputed_enc_kv,
                pad_idx=self.pad_idx,
            )

        return logits, attn_weights


class ViT_Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_len: int,
        dropout: float,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.encoder = _ViTEncoder(d_model, dropout)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
        )
        self.uniform = Uniform(0, 1)
        self.pad_idx = pad_idx

    def forward(self, imgs: Tensor, formulas: Tensor, epsilon: float = 1.0):
        batch_size = imgs.size(0)
        max_len = formulas.size(1)

        enc_out = self.encoder(imgs)
        precomputed_enc_kv = self.decoder.precompute_encoder_kv(enc_out)

        if self.training and epsilon < 0.9:
            logits_list = []
            prev_tokens = formulas[:, 0:1]

            for t in range(max_len):
                logits, attn_weights = self.decoder(
                    prev_tokens,
                    enc_out,
                    precomputed_enc_kv=precomputed_enc_kv,
                    pad_idx=self.pad_idx,
                )
                current_logit = logits[:, -1:, :]
                logits_list.append(current_logit)

                if t < max_len - 1:
                    if self.uniform.sample().item() < epsilon:
                        next_token = formulas[:, t + 1 : t + 2]
                    else:
                        next_token = torch.argmax(current_logit, dim=-1)
                    prev_tokens = torch.cat([prev_tokens, next_token], dim=1)

            logits = torch.cat(logits_list, dim=1)
        else:
            logits, attn_weights = self.decoder(
                formulas,
                enc_out,
                precomputed_enc_kv=precomputed_enc_kv,
                pad_idx=self.pad_idx,
            )

        return logits, attn_weights
