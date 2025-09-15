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
        # The forward pass is where we calculate the attention scores
        # First we get the batch size
        N = query.shape[0]
 
        # Then the sequence lengths
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Finally, pass inputs through our linear layers
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # To enable multi-head attention, we split our embedding into
        # self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # To get the raw attention score, we multiply Q with the
        # transpose of K ("energy") using einsum notation
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Now we divide it by sqrt(d_k), since this is multi-head
        scaled_energy = energy / (self.head_dim ** (1/2))

        # We'll also add a mask to support masked attention, if we want
        # it
        if mask:
            # Set masked positions to small values
            scaled_energy = scaled_energy.masked_fill(mask == 0,
                                                      float("-1e20"))
        
        # Now we can finally apply softmax to get attention weights that
        # look like probabilities (positive and sum to 1)
        attention = torch.softmax(scaled_energy, dim=-1)

        # Finally, multiply by V to get a weighted sum
        # Then we'll combine our heads back together and pass them
        # through the output layer
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

