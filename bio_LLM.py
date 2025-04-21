import torch
import torch.nn as nn
import torch.nn.functional as F
from LIF_neuron import compute_firing_rates

class BioAttentionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, mask=None):
        # Q, K, V: (B, H, T, D)
        B, H, T, D = Q.shape
        # device = Q.device
        
        Q_flat = Q.reshape(B*H*T, D)
        K_flat = K.reshape(B*H*T, D)
        
        # Calculate firing rates using CPU numpy function
        Q_np = Q_flat.detach().cpu().float().numpy()
        K_np = K_flat.detach().cpu().float().numpy()
        rates = compute_firing_rates(Q_np, K_np)  # (B*H*T,)
        
        # Convert rates back to torch tensor
        rates_t = torch.from_numpy(rates).to(Q).view(B, H, T)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            # Assuming mask is a lower triangular matrix of shape (1, 1, T, T)
            # Need to adapt it to our rate-based mechanism

            # For each position, zero out rates for tokens that shouldn't be attended to
            # causal_mask = torch.tril(torch.ones(T, T)).to(device)
            for t in range(T):
                # valid_positions = causal_mask[t].bool()
                # # Only normalize over valid positions
                denom = rates_t[:, :, :t+1].sum(dim=-1, keepdim=True) + 1e-9
                rates_t[:, :, :t+1] = rates_t[:, :, :t+1] / denom

                if t < T-1:
                    rates_t[:, :, t+1:] = 0
        else:
            # Normalize rates across sequence dimension (similar to softmax for now)
            # To do replace softmax with inhibitory neurons
            rates_t = rates_t / (rates_t.sum(dim=-1, keepdim=True) + 1e-9)
        
        ctx.save_for_backward(Q, K, V, rates_t)
        
        # Apply attention weights to values
        # Todo make this bio plausible too?
        rates_expanded = rates_t.view(B, H, T, 1)  # [B, H, T, 1]
        context = rates_expanded * V  # [B, H, T, D] = [B, H, T, 1] * [B, H, T, D]
        
        return context
    
    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, rates_t = ctx.saved_tensors
        B, H, T, D = Q.shape
        
        # Gradient wrt V (element-wise multiplication)
        rates_expanded = rates_t.view(B, H, T, 1)
        grad_V = rates_expanded * grad_output
        
        # Gradient wrt rates_t (from context = rates_t * V)
        grad_rates = (grad_output * V).sum(dim=-1)  # [B, H, T]
        
        # Simplified surrogate gradients for the LIF model
        # In a biological system, would need more complex gradient estimation
        # Todo replace?
        Q_norm = Q / (Q.norm(dim=-1, keepdim=True) + 1e-9)
        K_norm = K / (K.norm(dim=-1, keepdim=True) + 1e-9)
        
        grad_Q = grad_rates.unsqueeze(-1) * K_norm
        grad_K = grad_rates.unsqueeze(-1) * Q_norm
        
        return grad_Q, grad_K, grad_V, None  # None for mask

class BioSelfAttention(nn.Module):
    def __init__(self, n_heads, head_dim, block_size):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
    
    def forward(self, Q, K, V):
        return BioAttentionFn.apply(Q, K, V, self.mask)

class BioTransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        head_dim = n_embd // n_heads
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn = BioSelfAttention(n_heads, head_dim, block_size)
        
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Replace with more biologically plausible activation?
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

class BioTinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd=64, block_size=64, n_heads=4, n_layers=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.block_size = block_size
        
        self.blocks = nn.ModuleList([
            BioTransformerBlock(n_embd, n_heads, block_size) for _ in range(n_layers)
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