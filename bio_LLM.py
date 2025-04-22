import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LIF_neuron import Q_K_firing_rates, attention_V_firing_rates

def wta_inhibition(rates, steps=20, inhibition=-0.9, excitation=1.1):
    """
    rates: Tensor of shape (B, H, T, D) or (B, H, T)
    Returns: Tensor of the same shape after WTA inhibition and excitation is applied
    """
    orig_shape = rates.shape

    if len(rates.shape) == 4:
        B, H, T, D = rates.shape
        rates_flat = rates.reshape(B, H, T * D)
        W = torch.full((T * D, T * D), inhibition, device=rates.device)
    elif len(rates.shape) == 3:
        B, H, T = rates.shape
        rates_flat = rates
        W = torch.full((T, T), inhibition, device=rates.device)

    W.fill_diagonal_(excitation)

    for _ in range(steps):
        rates_flat = torch.clamp(rates_flat + torch.matmul(rates_flat, W.T), min=0.0, max=1.0)

    return rates_flat.reshape(orig_shape)


class BioSelfAttention(nn.Module):
    def __init__(self, n_heads, head_dim, block_size, 
                 LIF_model_dt, LIF_model_steps, 
                 wta_inhibition, wta_excitation, wta_steps):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.LIF_model_dt = LIF_model_dt
        self.LIF_model_steps = LIF_model_steps
        self.wta_inhibition = wta_inhibition
        self.wta_excitation = wta_excitation
        self.wta_steps = wta_steps
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, Q, K, V):
        B, H, T, D = Q.shape
        Q_flat = Q.reshape(B * H * T, D)
        K_flat = K.reshape(B * H * T, D)

        # Calculate firing rates using CPU numpy function
        Q_np = Q_flat.detach().cpu().float().numpy()
        K_np = K_flat.detach().cpu().float().numpy()

        rates = Q_K_firing_rates(Q_np, K_np, n_steps=self.LIF_model_steps, t_step=self.LIF_model_dt)  # (B*H*T,)

        # Convert rates back to torch tensor
        rates_t = torch.from_numpy(rates).to(Q).view(B, H, T)

        # WTA inhibition/excitation instead of softmax
        rates_t = wta_inhibition(rates_t, steps=self.wta_steps, 
                                 inhibition=self.wta_inhibition, 
                                 excitation=self.wta_excitation)
        
        # print(Q.shape)
        # print(K.shape)
        # print(V.shape)
        # print(rates_t.shape)

        B, H, T, D = V.shape
        V_flat = V.reshape(B * H * T, D)
        rates_t_flat = rates_t.reshape(B * H * T)

        rates_t_np = rates_t_flat.detach().cpu().float().numpy()
        V_np = V_flat.detach().cpu().float().numpy()
        
        # Apply attention weights to values via spiking readout
        context = attention_V_firing_rates(rates_t_np, V_np, n_steps=self.LIF_model_steps, t_step=self.LIF_model_dt)
        context = wta_inhibition(context, steps=self.wta_steps, 
                                 inhibition=self.wta_inhibition, 
                                 excitation=self.wta_excitation)

        return context

class BioTransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, 
                 LIF_model_dt, LIF_model_steps, 
                 wta_inhibition, wta_excitation, wta_steps):
        super().__init__()
        head_dim = n_embd // n_heads
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn = BioSelfAttention(n_heads, head_dim, block_size, 
                                     LIF_model_dt, LIF_model_steps, 
                                     wta_inhibition, wta_excitation, wta_steps)
        
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

class BioTinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd=64, block_size=64, n_heads=4, n_layers=4, 
                 LIF_model_dt=1e-2, LIF_model_steps=1000, 
                 wta_inhibition=-0.9, wta_excitation=1.1, wta_steps=20):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.block_size = block_size
        
        self.blocks = nn.ModuleList([
            BioTransformerBlock(n_embd, n_heads, block_size, 
                                LIF_model_dt, LIF_model_steps, 
                                wta_inhibition, wta_excitation, wta_steps) 
            for _ in range(n_layers)
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