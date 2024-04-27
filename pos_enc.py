import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        # initiating the encoding vector, max_length is the window size you would consider based on your text
        self.encoding = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # please refer to formula given for better understanding
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        # does not require encoding gradient since it won't change
        self.encoding = self.encoding.unsqueeze(0).requires_grad_(False)

    def forward(self, x):
        # Add positional encoding to each input sequence. Assume that x is of shape [batch_size, seq_length, d_model]
        return x + self.encoding[:, :x.size(1)]

