import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D,Summer

# LearnablePositionalEncoding
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_encoding = nn.Embedding(max_len, d_model)
        self.transformer_pos_encoding = PositionalEncoding1D(max_len)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        vit_pos_encoded = self.pos_encoding(positions)
        transformer_pos_encoded = self.transformer_pos_encoding(x)
        x = x + transformer_pos_encoded + vit_pos_encoded

        return x

