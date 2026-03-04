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
        """
        Args:
            d_model: latent vector size (embedding size)
            max_len: max sequence length
            dropout: dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a PE matrix with size (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate div_term (denominator) in log space for better precision
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin for even-indexed positions (2i)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos for odd-indexed positions (2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dim (1, max_len, d_model) for broadcasting when adding to embedding
        pe = pe.unsqueeze(0)

        # Register buffer so that it's not treated as learnable parameter,
        # However it will still be included in the model's state_dict
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor with dim (Batch_size, Seq_len, d_model)
        """
        # Adding PE to embedding input (cut according to the actual length of x)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
    

class Encoder(nn.Module):
    def __init__(
        self, 
        enc_out_dim: int, 
        dropout: float
    ) -> None:
        super().__init__()

        self.cnn_encoder = timm.create_model(
                    'resnet18', 
                    pretrained=True, 
                    features_only=True, 
                    out_indices=(4,),
                    output_stride=8
                )

        self.projection = nn.Conv2d(512, enc_out_dim, kernel_size=1) if enc_out_dim != 512 else nn.Identity()
        self.pos_encoder = PositionalEncoding(enc_out_dim, 10000, dropout=0)

    def forward(self, imgs: Tensor) -> Tensor:
        
        features = self.cnn_encoder(imgs)[0]  # [B, 512, H', W']
        features = self.projection(features)
        
        encoded_imgs = features.permute(0, 2, 3, 1)
        B, H, W, C = encoded_imgs.shape
        encoded_imgs = encoded_imgs.contiguous().view(B, H * W, C)

        encoded_imgs = self.pos_encoder(encoded_imgs)

        return encoded_imgs  # [B, L, 512], L = H*W
    

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        word_emb_dim: int,
        rnn_h_dim: int,
        rnn_o_dim: int,
        enc_out_dim: int,
        att_dim: int,
        dropout: float
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, word_emb_dim)
        self.rnn_decoder = nn.LSTMCell(rnn_o_dim + word_emb_dim, rnn_h_dim)
        self.dropout = nn.Dropout(dropout)

        # Init layers
        self.init_wh = nn.Linear(enc_out_dim, rnn_h_dim)
        self.init_wc = nn.Linear(enc_out_dim, rnn_h_dim)
        self.init_wo = nn.Linear(enc_out_dim, rnn_o_dim)

        # Attention layers
        self.beta = nn.Parameter(
            torch.Tensor(att_dim)
        )

        # Has a reasonable initial value (not too large, not too small). Helps the network learn attention scores stably from the start.
        init.uniform_(self.beta, -INIT_VALUE, INIT_VALUE)

        self.W_1 = nn.Linear(enc_out_dim, att_dim, bias=False)
        self.W_2 = nn.Linear(rnn_h_dim, att_dim, bias=False)
        self.W_3 = nn.Linear(rnn_h_dim + enc_out_dim, rnn_o_dim, bias=False)
        self.W_out = nn.Linear(rnn_o_dim, vocab_size, bias=False)

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
        enc_out: Tensor,
        projected_enc: Tensor,
        tgt: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor]:
        """Runing one step decoding"""

        # RNN stage
        prev_y = self.embedding(tgt).squeeze(1)  # prev_y's shape: [B, emb_size]
        inp = torch.cat([prev_y, o_t], dim=1)  # inp's shape: [B, emb_size + rnn_o_dim]
        h_t, c_t = self.rnn_decoder(inp, dec_states)  # h_t and c_t's shape: [B, rnn_h_dim]
        h_t = self.dropout(h_t)
        c_t = self.dropout(c_t)

        # Attention stage
        alpha = torch.tanh(
            projected_enc + self.W_2(h_t).unsqueeze(1)
        )  # [B, L, att_dim] + [B, 1, att_dim] -> [B, L, att_dim]
        alpha = torch.sum(self.beta * alpha, dim=-1)  # [B, L]
        alpha = F.softmax(alpha, dim=-1)  # [B, L]

        # calc context: [B, C]
        context_t = torch.bmm(
            alpha.unsqueeze(1), enc_out
        )  # batch mat mult [B, 1, L] @ [B, L, C] -> [B, 1, C]
        context_t = context_t.squeeze(1) # [B, 1, C] -> [B, C]

        # Output stage
        o_t = self.W_3(torch.cat([h_t, context_t], dim=1)).tanh()
        o_t = self.dropout(o_t)
        logit = self.W_out(o_t)  # [B, vocab_size]

        return (h_t, c_t), o_t, logit, alpha
    

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

        self.encoder = Encoder(enc_out_dim, dropout)
        self.decoder = Decoder(vocab_size, word_emb_dim, rnn_h_dim, rnn_o_dim, enc_out_dim, att_dim, dropout)
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

        # 3. Precompute for better performance, its shape: [B, L, att_dim]
        projected_encoder_output = self.decoder.W_1(enc_out)

        # 4. Loop to decode
        max_len = formulas.size(1)
        logits = []
        alphas = []
        prev_logit = None
        for t in range(max_len):
            tgt = formulas[:, t : t + 1]  # True label, shape: [B, 1]

            # Schedule sampling
            if prev_logit is not None and self.uniform.sample().item() > epsilon:
                tgt = torch.argmax(prev_logit.detach(), dim=1, keepdim=True) 

            # One step decoding
            dec_states, o_t, logit, alpha = self.decoder.forward_step(
                    dec_states, o_t, enc_out, projected_encoder_output, tgt
            )  # At each step returns 1 batch, logit: [B, vocab_size]
            
            prev_logit = logit.detach()
            logits.append(logit)
            alphas.append(alpha.detach().cpu())

        logits = torch.stack(logits, dim=1)  # [B, MAX_LEN, vocab_size]
        alphas = torch.stack(alphas, dim=1)  # [B, Max_Len, L]
        return logits, alphas