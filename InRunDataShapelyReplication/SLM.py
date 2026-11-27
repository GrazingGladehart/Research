from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    batch_size: int = 64
    block_size: int = 128
    max_iters: int = 4001
    eval_interval: int = 1000
    eval_iters: int = 100
    learning_rate: float = 1e-2
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2

    def __post_init__(self):
        if self.n_head <= 0 or self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by a positive n_head")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")


class Head(nn.Module):

    def __init__(self, head_size, config):
        super().__init__()

        # Transformer Head
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x)    # B,T,C
        q = self.query(x)  # B,T,C
        v = self.value(x)  # B,T,C

        headsize = k.size(-1)
        wei = q @ k.transpose(-2, -1) * headsize**-.5 # B,T,C @ B,C,T = B,T,T
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v # B,T,T @ B,T,C = B,T,C
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ Linear layer followed by nonlinearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config.n_head, head_size, config)
        self.ffwd = FeedForward(config.n_embd, config.dropout)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class SmallLanguageModel(nn.Module):

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config if config is not None else ModelConfig()

        # Read Data
        self.read_data()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.config.n_embd)
        self.position_embedding_table = nn.Embedding(self.config.block_size, self.config.n_embd)
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd) # final layer norm
        self.lm_head = nn.Linear(self.config.n_embd, self.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss=None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    
    def read_data(self):

        p = 'data/shakespeare.txt'
        print('Reading data from:', p)
        with open(p,'r',encoding='utf-8') as f:
            text = f.read()

        print('Character length of dataset:',len(text))

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # Create char,i encodings
        itoc = {i:c for i,c in enumerate(chars)}
        ctoi = {c:i for i,c in enumerate(chars)}
        self.encode = lambda x: [ctoi[c] for c in x]
        self.decode = lambda x: ''.join([itoc[i] for i in x])

        # Encode and Split Text
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9*len(data))
        self.train_data = data[:n]
        self.test_data = data[n:]



    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop to get block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def next_token(self,idx=None):

        if idx == None:
            idx = torch.zeros((1,1),dtype=torch.long, device=self.config.device)

        # crop to get block_size tokens
        idx_cond = idx[:, -self.config.block_size:]
        # get the predictions
        logits, loss = self(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # Add new char to context
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        # Convert inferred token to human-readable
        out = self.decode(idx_next[0].tolist())

        return idx, out