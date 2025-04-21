import torch

# Tiny Shakespeare dataset
with open("data/input.txt", "r", encoding="utf-8") as f:
    data = f.read()

chars = sorted(list(set(data)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data_tensor = torch.tensor(encode(data), dtype=torch.long)

def get_batch(block_size=64, batch_size=32, device='cpu'):
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

# Train/validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

train_tensor = torch.tensor(encode(train_data), dtype=torch.long)
val_tensor = torch.tensor(encode(val_data), dtype=torch.long)

def get_batch_split(split, block_size=64, batch_size=32, device='cpu'):
    dataset = train_tensor if split == 'train' else val_tensor
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i:i+block_size] for i in ix])
    y = torch.stack([dataset[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)