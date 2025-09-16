import math
import torch
import torch.nn as nn
from transformer_block import TransformerBlock

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length, # Max sentence length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # This is our embedding layer where we store word embeddings 
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

        # Pre-calculate the positional encoding matrix
        positional_encoding = torch.zeros(max_length, embed_size).to(device)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', positional_encoding)

        # This is where we can layer our transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        out = self.word_embedding(x)
        out += self.positional_encoding[:seq_length, :]
        out = self.dropout(out)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

