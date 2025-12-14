import torch
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from ..utils.config import *


class FormulaNet(VisionEncoderDecoderModel):
    def __init__(self, config):
        super().__init__(VisionEncoderDecoderConfig(**config))
        if ckpt_path := config.get("pretrained"):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.load_state_dict(state_dict, strict=True)
            # transformers.VisionEncoderDecoderModel is not smart enough to
            # initialize the model.config manually as the following.
            self.config.decoder_start_token_id = self.decoder.config.bos_token_id
            self.config.pad_token_id = self.decoder.config.pad_token_id
            self.config.eos_token_id = self.decoder.config.eos_token_id
