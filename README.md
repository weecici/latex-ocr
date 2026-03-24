# LaTeX OCR

An end-to-end deep learning system that converts images of mathematical formulas into LaTeX source code. Built as a comparative study of six encoder-decoder architectures — spanning CNN, ResNet, ViT encoders and LSTM, Transformer decoders — trained and evaluated on the **im2latex-100k** dataset.

---

## Team

| Student ID | Name               |
| :--------: | :----------------- |
|  23520199  | Cuong Nguyen       |
|  23520251  | Dat Le Thanh Thang |

---

## Architecture Overview

The project explores three tiers of model complexity, each progressively incorporating stronger attention mechanisms and modern vision backbones:

### Baseline — Additive Attention

| Model            | Encoder              | Decoder | Attention           |
| :--------------- | :------------------- | :------ | :------------------ |
| CNN + LSTM       | Custom 6-layer CNN   | LSTM    | Bahdanau (additive) |
| ResNet-18 + LSTM | Pretrained ResNet-18 | LSTM    | Bahdanau (additive) |

### Enhanced — Multi-Head Cross-Attention

| Model      | Encoder            | Decoder | Attention              |
| :--------- | :----------------- | :------ | :--------------------- |
| CNN + LSTM | Custom CNN + 2D PE | LSTM    | 8-head cross-attention |
| ViT + LSTM | ViT-Small/8        | LSTM    | 8-head cross-attention |

### Transformer — Full Transformer Decoder

| Model             | Encoder            | Decoder             | Attention               |
| :---------------- | :----------------- | :------------------ | :---------------------- |
| CNN + Transformer | Custom CNN + 2D PE | 8-layer Transformer | Multi-head self + cross |
| ViT + Transformer | ViT-Small/8        | 8-layer Transformer | Multi-head self + cross |

All models share a common vocabulary (`vocab.pkl`, extracted from im2latex-100k dataset) and use the same image preprocessing pipeline (pad to multiples of 16, grayscale normalization).

---

## Results

Evaluated on the im2latex-100k test set:

| Model                 |  BLEU-4   | Exact Match (%) | Edit Similarity (%) |
| :-------------------- | :-------: | :-------------: | :-----------------: |
| CNN + LSTM (Baseline) |   76.88   |      22.63      |        79.73        |
| ResNet-18 + LSTM      |   61.07   |      4.27       |        68.75        |
| CNN + LSTM (Enhanced) |   89.48   |      41.28      |        91.14        |
| ViT + LSTM            |   78.61   |      17.46      |        82.28        |
| **CNN + Transformer** | **91.46** |    **46.95**    |        93.87        |
| **ViT + Transformer** | **91.12** |    **46.06**    |      **93.91**      |

Key findings:

- Multi-head cross-attention and 2D Positional Encoding improved BLEU-4 by **+12.6** over the baseline CNN+LSTM
- The CNN with Transformer decoder achieved the highest overall scores, with **91.46 BLEU-4** and **46.95% exact match**
- Replacing the CNN encoder with ViT showed mixed results: helpful with the LSTM decoder tier but comparable in the Transformer Decoder tier

---

## Training Details

- **Dataset**: im2latex-100k — 100K image-formula pairs sourced from scientific papers
- **Optimizer**: Adam with linear warmup + ReduceLROnPlateau scheduling
- **Epochs**: 100
- **Label smoothing**: 0.1
- **Scheduled sampling**: linearly decayed teacher forcing ratio
- **Data augmentation**: affine transforms, perspective warp, Gaussian noise/blur, morphological operations, grid distortion, elastic transform
- **Batching**: smart bucket sampler to group images of similar spatial dimensions

---

## Web Demo

A unified Streamlit application (`app/app.py`) provides a single interface for all six models:

- **Model selector** — dropdown to switch between any of the six architectures
- **Beam search** controls (beam size 1-10, configurable max sequence length)
- **Editable LaTeX output** with live-rendered preview
- **Attention visualization** for LSTM-based models (baseline and enhanced)

### Running the demo

```bash
# Install dependencies
uv sync

# Download weights (see below) and place them in ./weights/

# Launch
uv run streamlit run app/ui.py
```

If you have [`just`](https://github.com/casey/just) installed, you can also run the app as follow:

```bash
uv sync
just
```

---

## Project Structure

```
latex-ocr/
├── app/
│   ├── app.py                  # Unified Streamlit application
│   ├── config.py               # Model registry & hyperparameters
│   ├── models/
│   │   ├── baseline.py         # CNN+LSTM, ResNet18+LSTM
│   │   ├── enhanced.py         # CNN+LSTM, ViT+LSTM (multi-head attention)
│   │   └── transformer.py      # CNN+Transformer, ViT+Transformer
│   ├── producers/
│   │   ├── baseline_producer.py
│   │   ├── enhanced_producer.py
│   │   └── transformer_producer.py
│   └── utils/
│       ├── vocab.py            # Vocabulary class
│       ├── transform.py        # Albumentations preprocessing
│       └── visualize.py        # Attention map visualization
├── notebooks/
│   ├── eda_preprocessing.ipynb
│   ├── data_aug_viz.ipynb
│   ├── baseline/
│   │   ├── cnn_lstm.ipynb
│   │   └── resnet18_lstm.ipynb
│   ├── enhanced/
│   │   ├── cnn_lstm.ipynb
│   │   └── vit_lstm.ipynb
│   └── transformer/
│       ├── cnn_transformer.ipynb
│       └── vit_transformer.ipynb
├── weights/                    # Model checkpoints (see below)
├── pyproject.toml
└── README.md
```

---

## Weights

Download the pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1-M25HwwNE1ChTGLpSqR6A-dPIHfPc5QL?usp=drive_link) and place them in the `weights/` directory:

```
weights/
├── vocab.pkl
├── cnn_lstm_baseline.pt
├── resnet18_lstm.pt
├── cnn_lstm_enhanced.pt
├── vit_lstm.pt
├── cnn_transformer.pt
└── vit_transformer.pt
```

---

## Tech Stack

Python | PyTorch | Streamlit | timm | Albumentations | OpenCV | Matplotlib | NLTK

---

## License

See [LICENSE](LICENSE) for details.

## References

[1] Deng, Y., Kanervisto, A., Ling, J., & Rush, A. M. (2017). Image-to-Markup Generation with Coarse-to-Fine Attention. https://arxiv.org/abs/1609.04938

[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023). Attention Is All You Need. https://arxiv.org/abs/1706.03762

[3] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. https://arxiv.org/abs/2010.11929
