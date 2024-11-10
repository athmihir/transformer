import torch

BATCH_SIZE = 4
CONTEXT_SIZE = 512
# gpt-4o tokenizer vocab size
VOCAB_SIZE = 200019
# size of embedding dimension
EMBEDDING_DIMENSION = 64
# device to train on
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# IMP: EMBEDDING_DIMENSION = HEAD_SIZE * ATTENTION_HEADS
# Size of single attention head
HEAD_SIZE = 8
# Number of Attention Heads
ATTENTION_HEADS = 8

# TRAINING VARIABLES

MAX_ITERS = 500
L_R = 1e-3
EVAL_INTERVAL = 100
EVAL_ITERS = 10



