import torch
from torch import nn
from torch.nn import functional as F
from constants import EMBEDDING_DIMENSION, VOCAB_SIZE, CONTEXT_SIZE, DEVICE, HEAD_SIZE, ATTENTION_HEADS


class Head(nn.Module):
    '''Single Head of self-attention'''
    
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(EMBEDDING_DIMENSION, HEAD_SIZE, bias=False)
        self.key = nn.Linear(EMBEDDING_DIMENSION, HEAD_SIZE, bias=False)
        self.value = nn.Linear(EMBEDDING_DIMENSION, HEAD_SIZE, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))
        

    def forward(self, x):
        K = self.key(x) # (B, CONTEXT_SIZE, HEAD_SIZE)
        Q = self.query(x) # (B, CONTEXT_SIZE, HEAD_SIZE)
        V = self.value(x) # (B, CONTEXT_SIZE, HEAD_SIZE)
        wei = Q @ K.transpose(-2, -1) * HEAD_SIZE**-0.5 # (B, CONTEXT_SIZE, CONTEXT_SIZE)
        wei = wei.masked_fill(self.tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ V # (B, CONTEXT_SIZE, HEAD_SIZE)
        return out

class MultiHeadAttention(nn.Module):
    '''multiple heads of self-attention in parallel'''

    def __init__(self, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(num_heads)])
        self.proj = nn.Linear(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    '''linear layer followed by a layer of non-linearity'''

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    '''Transformer Block: communication followed by computation'''

    def __init__(self):
        super().__init__()
        self.sa_head = MultiHeadAttention(ATTENTION_HEADS)
        self.ffwd = FeedForward(EMBEDDING_DIMENSION)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x

class Transformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION)
        self.position_embedding_table = nn.Embedding(CONTEXT_SIZE, EMBEDDING_DIMENSION)
        self.blocks = nn.Sequential(
            Block(),
            Block(),
            Block(),
        )
        self.lm_head = nn.Linear(EMBEDDING_DIMENSION, VOCAB_SIZE)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,CONTEXT_SIZE,EMBEDDING_DIMENSION)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (CONTEXT_SIZE, EMBEDDING_DIMENSION)
        input_embed = tok_emb + pos_emb # (B, CONTEXT_SIZE, EMBEDDING_DIMENSION)
        x = self.blocks(input_embed) # (B, CONTEXT_SIZE, EMBEDDING_DIMENSION)
        logits = self.lm_head(x) # (B, CONTEXT_SIZE, VOCAB_SIZE)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -CONTEXT_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # Becomes (B, C)
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

