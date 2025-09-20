import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0.1,
        device="cuda",
        max_length=100,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # Create a boolean mask where True indicates a padding token (a position to discard)
        mask_to_discard = (src == self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    
        # Create a float16 tensor of zeros on the same device
        float_mask = torch.zeros_like(
            mask_to_discard, dtype=torch.float
        )
        
        # Where the discard mask is True, fill the float mask with -inf
        return float_mask.masked_fill(mask_to_discard, float("-inf"))

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        
        # Create padding mask (True == padding token)
        trg_pad_mask = (trg == self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Create subsequent mask (True == future token)
        trg_sub_mask = torch.triu(
            torch.ones((trg_len, trg_len), device=self.device), diagonal=1
        ).bool()
        
        # Combine them: a position is discarded if it's padding OR a future token
        mask_to_discard = trg_pad_mask | trg_sub_mask
        
        # Convert the final boolean mask to a float mask
        float_mask = torch.zeros_like(
            mask_to_discard, dtype=torch.float
        )
    
        return float_mask.masked_fill(mask_to_discard, float("-inf"))

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        num_heads = self.encoder.layers[0].attention.heads

        # Expand the source mask for the encoder's self attention
        src_mask_repeated = src_mask.repeat(1, num_heads, src.size(1), 1)
        trg_mask_repeated = trg_mask.repeat(1, num_heads, 1, 1)
        
        enc_src = self.encoder(src, src_mask_repeated)
        out = self.decoder(trg, enc_src, src_mask_repeated, trg_mask_repeated)

        return out
