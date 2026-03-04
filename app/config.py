"""
Unified configuration for all LaTeX OCR models.

Each model is registered in MODEL_REGISTRY with:
  - weight:       path to the .pt checkpoint (relative to project root)
  - category:     Baseline | Enhanced | Transformer
  - description:  one-line summary shown in the sidebar
  - build_fn:     callable(vocab_size) -> nn.Module
  - producer_cls: the LatexProducer class to use for inference
  - has_attention: whether the producer returns attention maps
"""

import os
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
VOCAB_PATH = os.path.join(WEIGHTS_DIR, "vocab.pkl")

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_BEAM_SIZE = 5
DEFAULT_MAX_LEN = 150


# ── Model builder helpers ────────────────────────────────────────────────────
# Baseline
def _build_baseline_cnn_lstm(vocab_size: int):
    from models.baseline import CNN_LSTM

    return CNN_LSTM(
        vocab_size,
        word_emb_dim=80,
        rnn_h_dim=512,
        rnn_o_dim=512,
        enc_out_dim=512,
        att_dim=512,
        dropout=0.2,
    )


def _build_baseline_resnet18_lstm(vocab_size: int):
    from models.baseline import ResNet18_LSTM

    return ResNet18_LSTM(
        vocab_size,
        word_emb_dim=80,
        rnn_h_dim=512,
        rnn_o_dim=512,
        enc_out_dim=512,
        att_dim=512,
        dropout=0.2,
    )


# Enhanced
def _build_enhanced_cnn_lstm(vocab_size: int):
    from models.enhanced import CNN_LSTM

    return CNN_LSTM(
        vocab_size,
        word_emb_dim=80,
        rnn_h_dim=512,
        rnn_o_dim=512,
        enc_out_dim=512,
        att_dim=512,
        n_heads=8,
        dropout=0,
    )


def _build_enhanced_vit_lstm(vocab_size: int):
    from models.enhanced import ViT_LSTM

    return ViT_LSTM(
        vocab_size,
        word_emb_dim=80,
        rnn_h_dim=512,
        rnn_o_dim=512,
        enc_out_dim=512,
        att_dim=512,
        n_heads=8,
        dropout=0,
    )


# Transformer
def _build_cnn_transformer(vocab_size: int):
    from models.transformer import CNN_Transformer

    return CNN_Transformer(
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
        max_len=151,
        dropout=0,
        pad_idx=1,
    )


def _build_vit_transformer(vocab_size: int):
    from models.transformer import ViT_Transformer

    return ViT_Transformer(
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
        max_len=151,
        dropout=0,
        pad_idx=1,
    )


# ── Producer imports (deferred to avoid heavy imports at config-parse time) ──
def _baseline_producer():
    from producers.baseline_producer import LatexProducer

    return LatexProducer


def _enhanced_producer():
    from producers.enhanced_producer import LatexProducer

    return LatexProducer


def _transformer_producer():
    from producers.transformer_producer import LatexProducer

    return LatexProducer


# ── Registry ─────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "CNN + LSTM (Baseline)": {
        "weight": os.path.join(WEIGHTS_DIR, "cnn_lstm_baseline.pt"),
        "category": "Baseline",
        "description": "CNN encoder with LSTM decoder and additive attention.",
        "build_fn": _build_baseline_cnn_lstm,
        "producer_cls": _baseline_producer(),
        "has_attention": True,
    },
    "ResNet-18 + LSTM": {
        "weight": os.path.join(WEIGHTS_DIR, "resnet18_lstm.pt"),
        "category": "Baseline",
        "description": "Pretrained ResNet-18 encoder with LSTM decoder.",
        "build_fn": _build_baseline_resnet18_lstm,
        "producer_cls": _baseline_producer(),
        "has_attention": True,
    },
    "CNN + LSTM (Enhanced)": {
        "weight": os.path.join(WEIGHTS_DIR, "cnn_lstm_enhanced.pt"),
        "category": "Enhanced",
        "description": "CNN encoder with multi-head cross-attention LSTM decoder.",
        "build_fn": _build_enhanced_cnn_lstm,
        "producer_cls": _enhanced_producer(),
        "has_attention": True,
    },
    "ViT + LSTM": {
        "weight": os.path.join(WEIGHTS_DIR, "vit_lstm.pt"),
        "category": "Enhanced",
        "description": "ViT encoder with multi-head cross-attention LSTM decoder.",
        "build_fn": _build_enhanced_vit_lstm,
        "producer_cls": _enhanced_producer(),
        "has_attention": True,
    },
    "CNN + Transformer": {
        "weight": os.path.join(WEIGHTS_DIR, "cnn_transformer.pt"),
        "category": "Transformer",
        "description": "CNN encoder with Transformer decoder and KV-cache beam search.",
        "build_fn": _build_cnn_transformer,
        "producer_cls": _transformer_producer(),
        "has_attention": False,
    },
    "ViT + Transformer": {
        "weight": os.path.join(WEIGHTS_DIR, "vit_transformer.pt"),
        "category": "Transformer",
        "description": "ViT encoder with Transformer decoder and KV-cache beam search.",
        "build_fn": _build_vit_transformer,
        "producer_cls": _transformer_producer(),
        "has_attention": False,
    },
}
