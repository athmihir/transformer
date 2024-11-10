from model import Transformer
from data_loader import DataLoader
import tiktoken
import torch
from constants import DEVICE, MAX_ITERS, L_R, EVAL_INTERVAL, EVAL_ITERS
import os

enc = tiktoken.encoding_for_model("gpt-4o")

@torch.no_grad()
def estimate_loss(model, data_loader):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = data_loader.get_next_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("Loading data...")
# Load state if it exists
dl = None
if os.path.exists('data/dl_state.pkl'):
    print("Found existing data loader state, booting ...")
    dl = DataLoader('data/input_tensor1.pt', 'data/dl_state.pkl')
else:
    dl = DataLoader('data/input_tensor1.pt')

print("Initializing model...")
model = Transformer().to(DEVICE)
# Load if model already exists
if os.path.exists('data/model.pt'):
    print('Found a previous model, re-using that...')
    model.load_state_dict(torch.load('data/model.pt', weights_only=True))
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=L_R)

print("Beginning training phase...")
for iter in range(MAX_ITERS):

    # occasionally evaluate the loss on train set
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS -1:
        losses = estimate_loss(model, dl)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = dl.get_next_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"step: {iter}, loss: {loss.item()}")

print("Saving the model...")
torch.save(model.state_dict(), 'data/model.pt')

print("Saving the state of data loader...")
dl.save_state('data/dl_state.pkl')

print("Making inference...")
print(enc.decode(model.generate(dl.get_next_batch('train')[0], 100)[0].tolist()))


