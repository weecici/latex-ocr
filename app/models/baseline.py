"""
Baseline models: CNN+LSTM and ResNet18+LSTM with additive (Bahdanau) attention.
"""

import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.distributions.uniform import Uniform
from torch.nn import init
import torch.nn.functional as F
import timm

INIT_VALUE = 1e-2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# CNN Encoder (for CNN+LSTM baseline)
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

        self.pos_encoder = PositionalEncoding(enc_out_dim, 10000, dropout=0)

    def forward(self, imgs: Tensor) -> Tensor:
        encoded_imgs = self.cnn_encoder(imgs)
        encoded_imgs = encoded_imgs.permute(0, 2, 3, 1)

        B, H, W, C = encoded_imgs.shape
        encoded_imgs = encoded_imgs.contiguous().view(B, H * W, -1)
        encoded_imgs = self.pos_encoder(encoded_imgs)
        return encoded_imgs


# ---------------------------------------------------------------------------
# ResNet-18 Encoder
# ---------------------------------------------------------------------------
class _ResNet18Encoder(nn.Module):
    def __init__(self, enc_out_dim: int, dropout: float) -> None:
        super().__init__()

        self.cnn_encoder = timm.create_model(
            "resnet18",
            pretrained=True,
            features_only=True,
            out_indices=(4,),
            output_stride=8,
        )

        self.projection = (
            nn.Conv2d(512, enc_out_dim, kernel_size=1)
            if enc_out_dim != 512
            else nn.Identity()
        )
        self.pos_encoder = PositionalEncoding(enc_out_dim, 10000, dropout=0)

    def forward(self, imgs: Tensor) -> Tensor:
        features = self.cnn_encoder(imgs)[0]
        features = self.projection(features)

        encoded_imgs = features.permute(0, 2, 3, 1)
        B, H, W, C = encoded_imgs.shape
        encoded_imgs = encoded_imgs.contiguous().view(B, H * W, C)
        encoded_imgs = self.pos_encoder(encoded_imgs)
        return encoded_imgs


# ---------------------------------------------------------------------------
# Shared Decoder (additive / Bahdanau attention)
# ---------------------------------------------------------------------------
class _BaselineDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_emb_dim: int,
        rnn_h_dim: int,
        rnn_o_dim: int,
        enc_out_dim: int,
        att_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, word_emb_dim)
        self.rnn_decoder = nn.LSTMCell(rnn_o_dim + word_emb_dim, rnn_h_dim)
        self.dropout = nn.Dropout(dropout)

        self.init_wh = nn.Linear(enc_out_dim, rnn_h_dim)
        self.init_wc = nn.Linear(enc_out_dim, rnn_h_dim)
        self.init_wo = nn.Linear(enc_out_dim, rnn_o_dim)

        self.beta = nn.Parameter(torch.Tensor(att_dim))
        init.uniform_(self.beta, -INIT_VALUE, INIT_VALUE)

        self.W_1 = nn.Linear(enc_out_dim, att_dim, bias=False)
        self.W_2 = nn.Linear(rnn_h_dim, att_dim, bias=False)
        self.W_3 = nn.Linear(rnn_h_dim + enc_out_dim, rnn_o_dim, bias=False)
        self.W_out = nn.Linear(rnn_o_dim, vocab_size, bias=False)

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
        enc_out: Tensor,
        projected_enc: Tensor,
        tgt: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor, Tensor]:
        prev_y = self.embedding(tgt).squeeze(1)
        inp = torch.cat([prev_y, o_t], dim=1)
        h_t, c_t = self.rnn_decoder(inp, dec_states)
        h_t = self.dropout(h_t)
        c_t = self.dropout(c_t)

        alpha = torch.tanh(projected_enc + self.W_2(h_t).unsqueeze(1))
        alpha = torch.sum(self.beta * alpha, dim=-1)
        alpha = F.softmax(alpha, dim=-1)

        context_t = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)

        o_t = self.W_3(torch.cat([h_t, context_t], dim=1)).tanh()
        o_t = self.dropout(o_t)
        logit = self.W_out(o_t)

        return (h_t, c_t), o_t, logit, alpha


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
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = _CNNEncoder(enc_out_dim, dropout)
        self.decoder = _BaselineDecoder(
            vocab_size,
            word_emb_dim,
            rnn_h_dim,
            rnn_o_dim,
            enc_out_dim,
            att_dim,
            dropout,
        )
        self.uniform = Uniform(0, 1)

    def forward(self, imgs: Tensor, formulas: Tensor, epsilon: float) -> Tensor:
        enc_out = self.encoder(imgs)
        dec_states, o_t = self.decoder.init_decoder_states(enc_out)
        projected_encoder_output = self.decoder.W_1(enc_out)

        max_len = formulas.size(1)
        logits = []
        alphas = []
        prev_logit = None
        for t in range(max_len):
            tgt = formulas[:, t : t + 1]
            if prev_logit is not None and self.uniform.sample().item() > epsilon:
                tgt = torch.argmax(prev_logit.detach(), dim=1, keepdim=True)
            dec_states, o_t, logit, alpha = self.decoder.forward_step(
                dec_states, o_t, enc_out, projected_encoder_output, tgt
            )
            prev_logit = logit.detach()
            logits.append(logit)
            alphas.append(alpha.detach().cpu())

        logits = torch.stack(logits, dim=1)
        alphas = torch.stack(alphas, dim=1)
        return logits, alphas


class ResNet18_LSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_emb_dim: int,
        rnn_h_dim: int,
        rnn_o_dim: int,
        enc_out_dim: int,
        att_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = _ResNet18Encoder(enc_out_dim, dropout)
        self.decoder = _BaselineDecoder(
            vocab_size,
            word_emb_dim,
            rnn_h_dim,
            rnn_o_dim,
            enc_out_dim,
            att_dim,
            dropout,
        )
        self.uniform = Uniform(0, 1)

    def forward(self, imgs: Tensor, formulas: Tensor, epsilon: float) -> Tensor:
        enc_out = self.encoder(imgs)
        dec_states, o_t = self.decoder.init_decoder_states(enc_out)
        projected_encoder_output = self.decoder.W_1(enc_out)

        max_len = formulas.size(1)
        logits = []
        alphas = []
        prev_logit = None
        for t in range(max_len):
            tgt = formulas[:, t : t + 1]
            if prev_logit is not None and self.uniform.sample().item() > epsilon:
                tgt = torch.argmax(prev_logit.detach(), dim=1, keepdim=True)
            dec_states, o_t, logit, alpha = self.decoder.forward_step(
                dec_states, o_t, enc_out, projected_encoder_output, tgt
            )
            prev_logit = logit.detach()
            logits.append(logit)
            alphas.append(alpha.detach().cpu())

        logits = torch.stack(logits, dim=1)
        alphas = torch.stack(alphas, dim=1)
        return logits, alphas
