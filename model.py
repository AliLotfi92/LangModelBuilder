import torch
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn 

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes, num_heads, num_layer) -> None:
        super(LanguageModel, self).__init__()
        # the size of vocab_size of your trained tokenizer, you can get it by tokenizer.get_vocab()
        self.vocab_size = vocab_size
        # the embeddding dimension for each token, for a model like bert it is 768
        self.embed_dim = embed_dim
        # number of classes for text classification
        self.num_classes = num_classes
        # number of heads for multi-head attetion layer
        self.num_heads = num_heads
        # number of repeated encoder layer for a base bert model it is 12
        self.num_layer = num_layer
        # ignore idex for unmasked tokens
        self.ignore_index = -100
        
        # decoder layer
        self.fc = nn.Linear(embed_dim, vocab_size)
        

