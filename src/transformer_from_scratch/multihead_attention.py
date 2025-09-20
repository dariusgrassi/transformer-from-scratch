import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # We must ensure that the embedding size can be split evenly
        # among the attention heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size must be divisible by heads!"

        self.values = nn.Linear(self.embed_size, self.embed_size,
                                bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size,
                                bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size,
                                bias=False)
        # This is the output layer
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Reshape for multi-head attention
        values = self.values(values).reshape(N, value_len, self.heads, self.head_dim)
        keys = self.keys(keys).reshape(N, key_len, self.heads, self.head_dim)
        queries = self.queries(query).reshape(N, query_len, self.heads, self.head_dim)

        # --- Standard PyTorch Attention Logic ---
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        scaled_energy = energy / (self.embed_size ** (1/2))

        # The mask is now a float tensor (0.0 or -inf), so we add it directly
        if mask is not None:
            scaled_energy += mask # <<< This is the correct logic

        attention = torch.softmax(scaled_energy, dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # --- End of Standard Logic ---

        out = self.fc_out(out)
        return out

