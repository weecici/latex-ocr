from texo.data.processor import EvalMERImageProcessor
from texo.model.formulanet import FormulaNet
from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
    PreTrainedTokenizerFast,
)
from PIL import Image
import IPython
import torch
import torchinfo
import pickle as pkl


# def load(path):
#     tokenizer = AutoTokenizer.from_pretrained(path)
#     model = VisionEncoderDecoderModel.from_pretrained(path)

#     return model, tokenizer


# def inference(
#     model: FormulaNet, image_path: str, tokenizer: PreTrainedTokenizerFast, device
# ):
#     model.to(device)
#     image = Image.open(image_path)
#     image_processor = EvalMERImageProcessor(image_size={"width": 384, "height": 384})
#     processed_image = image_processor(image).unsqueeze(0)
#     outputs = model.generate(pixel_values=processed_image.to(device))
#     pred_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
#     return pred_str


# if __name__ == "__main__":
#     model, tokenizer = load("alephpi/FormulaNet")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     image_path = "./data/test_img/a.png"
#     pred_str = inference(model, image_path, tokenizer, device)
#     print(pred_str)

from crnn import Im2LatexModel, Config, Vocab, inference


def load_vocab(filepath: str) -> Vocab:
    with open(filepath, "rb") as f:
        vocab: Vocab = pkl.load(f)
    print(f"Load vocab including {len(vocab)} Latex tokens.")
    return vocab


args = Config()
vocab = load_vocab(args.vocab_path)
vocab_size = len(vocab)
# print(vocab.sign2id)

model = Im2LatexModel(
    vocab_size,
    args.word_emb_dim,
    args.rnn_h_dim,
    args.rnn_o_dim,
    args.enc_out_dim,
    args.att_dim,
    args.dropout,
)

model_path = "models/crnn.pt"
ckpt = torch.load(model_path, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

print(inference(model, vocab, "data/test_img/e.png", torch.device("cuda")))
