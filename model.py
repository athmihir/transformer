import torch
from torch import nn
from torch.nn import functional as F

class Transformer(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = logits.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

