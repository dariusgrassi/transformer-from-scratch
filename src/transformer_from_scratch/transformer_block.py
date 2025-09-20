import torch
import torch.nn as nn
from .multihead_attention import SelfAttention

# With the attention block defined, we can now build the Transformer
# block
# Input will first come into the SelfAttention module
# Then we will use the skip connection, to mitigate vanishing gradients
# Next, Layer Normalization, to aid in generalization
# We will then send the result into a Feed-Forward network
# Add another skip connection, and apply another Layer Normalization

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        # The transformer encoder architecture
        # Self attention, layer normalization, and a feed-forward
        # network
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Self attention input sequence
        # Multi-head attention block
        # Then skip connection, normalization, and dropout
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))

        # Feed forward input sequence
        # Feed forward block, then skip connection (notice add x, not query)
        # Then normalize and dropout
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

