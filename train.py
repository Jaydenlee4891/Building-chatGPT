import torch
import torch.nn as nn
from torch.nn import functional as f

!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding ='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtpye=torch.long)
n = int(0.9*len(data))
train_data=data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data= train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x= torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)
print('-------')

for b in range(batch_size):
    for t in range(block_size):
        context=xb[b, :t+1]

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self,idx,targets):
        logits = self.token_embedding_table(idx)

        return logits

        target= yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")
