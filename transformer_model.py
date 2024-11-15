import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import math

# transformer model - encoder only
class LMTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_heads, num_layers):
        super(LMTransformer, self).__init__()
        self.vocab_size = vocab_size  # vocab size for embedding matrix
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer from vocab size to embedding dimension
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.1)  # positional encoding defined below
        ignore_index = -100  # ignore index for those tokens that have not been masked
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)  # loss function for masked tokens, it's a CE-based loss
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=num_heads, batch_first=True)  # each encoder block
        self.transformers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # repeating the encoder block for number of layers
        self.fc = nn.Linear(embed_dim, num_classes)  # last dense layer that goes from embedding dim to number of classes, which is here number of vocabs

    # forward loop
    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None, return_outputs=False):
        emb = self.embedding(input_ids)  # input embeddings
        emb_pos = self.pos_encoder(emb)  # positional encoding
        attention_mask = (attention_mask == 0)  # attention mask
        trans_out = self.transformers(emb_pos, src_key_padding_mask=attention_mask)  # transformer layers
        logits = self.fc(trans_out)  # output layer
        if return_outputs:
            return logits, trans_out
        loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1)).unsqueeze(0)
        return loss
            

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.0, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = embed_dim
        self.max_len = max_len

        pe = torch.zeros(self.max_len, embed_dim)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
