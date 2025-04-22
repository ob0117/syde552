import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LIFCollection:
    def __init__(self, n=1, dim=1, tau_rc=0.02, tau_ref=0.002, v_th=1,
                 max_rates=(200, 400), intercept_range=(-1, 1),
                 t_step=0.001, v_init=0.0):
        self.n = n
        self.dim = dim
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self.v_th = np.ones(n) * v_th
        self.t_step = t_step

        self.voltage = np.ones(n) * v_init
        self.refractory_time = np.zeros(n)
        self.output = np.zeros(n)

        max_rates_tensor = np.random.uniform(*max_rates, n)
        intercepts_tensor = np.random.uniform(*intercept_range, n)

        self.gain = (
            self.v_th *
            (1 - 1 / (1 - np.exp((self.tau_ref - 1/max_rates_tensor) /
                                 self.tau_rc))) /
            (intercepts_tensor - 1)
        )
        self.bias = np.expand_dims(
            self.v_th - self.gain * intercepts_tensor, axis=1
        )

        self.encoders = np.random.randn(n, self.dim)
        self.encoders /= np.linalg.norm(self.encoders, axis=1)[:, None]

    def reset(self):
        self.voltage[:] = 0.0
        self.refractory_time[:] = 0.0
        self.output[:] = 0.0

    def step(self, inputs):
        dt = self.t_step
        self.refractory_time -= dt
        delta_t = np.clip(dt - self.refractory_time, 0.0, dt)

        I = np.sum(self.bias + inputs * self.encoders * self.gain[:, None],
                   axis=1)

        self.voltage = I + (self.voltage - I) * np.exp(-delta_t / self.tau_rc)

        spike_mask = self.voltage > self.v_th
        self.output[:] = spike_mask / dt

        t_spike = (
            self.tau_rc *
            np.log((self.voltage[spike_mask] - I[spike_mask]) /
                   (self.v_th[spike_mask] - I[spike_mask])) +
            dt
        )
        self.voltage[spike_mask] = 0.0
        self.refractory_time[spike_mask] = self.tau_ref + t_spike

        return self.output
def lif_rate(I, tau_rc=0.02, tau_ref=0.002, v_th=1.0, eps=1e-7):
    good = I > v_th + eps
    r = torch.zeros_like(I)
    log_term = torch.log1p(-v_th / I[good])           # log(1 - V_th/I)
    r[good] = 1.0 / (tau_ref - tau_rc * log_term)
    return r     

def compute_firing_rates(Q, K,
                         n_steps=1000, t_step=1e-3,
                         lif_kwargs=None, seed=None):
    if Q.ndim != 2 or K.ndim != 2:
        raise ValueError("Q and K must be two‑dimensional")
    if Q.shape[1] != K.shape[1]:
        raise ValueError("Dimensionality mismatch between Q and K")
    if seed is not None:
        np.random.seed(seed)

    lif_kwargs = lif_kwargs or {}
    d = Q.shape[1]
    n_k = K.shape[0]

    # Normalise encoders once
    enc = K.astype(float).copy()
    enc /= np.linalg.norm(enc, axis=1, keepdims=True) + 1e-12

    rates = np.zeros((Q.shape[0], n_k), dtype=float)
    duration = n_steps * t_step

    for qi, q_vec in enumerate(Q):
        # --- create a brand‑new LIF population so states don't leak ---
        lif = LIFCollection(n=n_k, dim=d, t_step=t_step, **lif_kwargs)
        lif.encoders = enc

        # replicate the query so every neuron receives the same stimulus
        inp = np.repeat(q_vec[None, :], n_k, axis=0)

        spikes = np.zeros(n_k, dtype=float)
        for _ in range(n_steps):
            spikes += lif.step(inp) * t_step
        rates[qi] = spikes / duration

    return rates


class StandardSelfAttention(nn.Module):
    def __init__(self, n_heads, head_dim, block_size):
        super().__init__()
        self.head_dim = head_dim
        self.temperature = head_dim ** 0.5

        enc = torch.randn(n_heads, block_size, head_dim)
        self.register_buffer("enc_hat", F.normalize(enc, dim=-1, eps=1e-8))

        self.register_buffer("gain",
            torch.rand(1, n_heads, block_size, 1) * 200)
        self.register_buffer("bias",
            torch.rand(1, n_heads, block_size, 1))

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, Q, K, V):                       # K unused, kept for API
        B, H, T, D = Q.shape
        E = self.enc_hat[:, :T, :]                    # (H, T, D)
        G = self.gain[:, :, :T, :]                    # (1, H, T, 1)
        BIAS = self.bias[:, :, :T, :]                 # (1, H, T, 1)

        I = G * torch.einsum("b h t d, h s d -> b h t s", Q, E) + BIAS
        attn_scores = lif_rate(I) / self.temperature
        attn_scores = attn_scores.masked_fill(
            self.mask[:, :, :T, :T] == 0, -float('inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
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