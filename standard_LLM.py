import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardSelfAttention(nn.Module):
    def __init__(self, n_heads, head_dim, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.temperature = head_dim ** 0.5
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, Q, K, V):
        B, H, T, D = Q.shape
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.temperature
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        head_dim = n_embd // n_heads
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn = StandardSelfAttention(n_heads, head_dim, block_size)
        
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        
        self.n_heads = n_heads
        self.head_dim = head_dim

    def forward(self, x):
        B, T, C = x.shape
        
        x_norm = self.ln1(x)
        qkv = self.qkv_proj(x_norm)
        
        # Multi-head attention
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # Each is (B, H, T, D)
        
        attn_out = self.attn(Q, K, V)  # (B, H, T, D)
        attn_out = attn_out.transpose(1, 2).contiguous()  # (B, T, H, D)
        attn_out = attn_out.view(B, T, self.n_heads * self.head_dim)  # (B, T, C)
        attn_out = self.out_proj(attn_out)
        
        x = x + attn_out
        
        x_norm = self.ln2(x)
        x = x + self.mlp(x_norm)
        
        return x

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd=64, block_size=64, n_heads=4, n_layers=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.block_size = block_size
        
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_heads, block_size) for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.size()
        assert T <= self.block_size, f"Input sequence length ({T}) exceeds block size ({self.block_size})"
        
        x = self.token_embedding(x)
        x = x + self.pos_embedding[:, :T, :]
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_final(x)
        logits = self.head(x)
        
        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text from the model."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx