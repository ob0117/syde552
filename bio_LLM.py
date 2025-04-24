import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DOT_MODE_STANDARD     = "STANDARD"      # default
DOT_MODE_NEURON       = "NEURON_DOT"

SOFTMAX_MODE_STANDARD = "STANDARD"      # default
SOFTMAX_MODE_NEURON   = "NEURON_SOFTMAX"


# ------- LIF neuron methods ----------------------------------------------------------------------------------------

def lif_rate(I, tau_rc=0.02, tau_ref=0.002, v_th=0.01, eps=1e-7):
    good = I > v_th + eps
    r = torch.zeros_like(I)
    log_term = torch.log1p(-v_th / I[good])           # log(1 - V_th/I)
    r[good] = 1.0 / (tau_ref - tau_rc * log_term)
    r = torch.nan_to_num(r, nan=0.0, posinf=float('inf'), neginf=0.0)
    return r 

def divisive_softmax(rates, mask, sigma=1e-3):
    """
    rates : tensor ⟨B,H,T,S⟩ raw firing rate drive D
    sigma  : Prevents division by zero and sets the contrast at which inhibition starts to bite. If sigma big,
    even a busy pool only hardly suppresses responses; if sigma=0 the strongest spikes win a winner-take-all.
    return : R_j   = divisively normalised rates
    returns -> divisively normalised weights R
    """
    rates = torch.where(mask, rates, torch.zeros_like(rates))  # -inf → 0
    rates = torch.clamp_min(rates, 0.0)                       # just in case

    pool  = rates.pow(2).sum(-1, keepdim=True).sqrt()         # √Σ_k D_k²
    weights = rates / (sigma + pool)

    weights = torch.where(mask, weights, torch.zeros_like(weights))

    return weights 


# ------- Tiny LLM classes -------------------------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, n_heads, head_dim, block_size, dot_mode: str = DOT_MODE_STANDARD,
                 softmax_mode: str = SOFTMAX_MODE_STANDARD):
        super().__init__()
        self.dot_mode     = dot_mode
        self.softmax_mode = softmax_mode
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

    def _lif_similarity(self, Q, K):
        K_hat = F.normalize(K, dim=-1, eps=1e-8)               # (B,H,S,D)
        dot   = torch.einsum("b h t d, b h s d -> b h t s", Q, K_hat)

        S = K.size(2)                                          # current length
        gain = self.gain[:, :, :S, :]                          # ★ slice ★
        bias = self.bias[:, :, :S, :]                          # ★ slice ★

        I = gain * dot + bias
        return lif_rate(I) / self.temperature 

    def forward(self, Q, K, V):
        B, H, T, D = Q.shape

        # ---------- similarity ------------------------------------
        if self.dot_mode == "NEURON_DOT":
            attn_scores = self._lif_similarity(Q, K)
        else:                                           # STANDARD
            attn_scores = torch.matmul(
                Q, K.transpose(-2, -1)) / self.temperature

        # ---------- softmax ---------------------------------------
        if self.softmax_mode == "NEURON_SOFTMAX":
            if self.dot_mode != DOT_MODE_NEURON:
                raise ValueError("NEURON_SOFTMAX needs NEURON_DOT similarity")
            valid = (self.mask[:, :, :T, :T] == 1)
            attn_weights = divisive_softmax(attn_scores, valid)
        else:
            # causal mask (conventionally applied, but doesn't make sense for the bio method)
            attn_scores_mask = attn_scores.masked_fill(
                self.mask[:, :, :T, :T] == 0, -float('inf'))
            attn_weights = torch.softmax(attn_scores_mask, dim=-1)
        
        self._last_attn_scores = attn_scores.detach()
        self._last_attn_weights = attn_weights.detach()

        # ---------- value projection ------------------------------
        return torch.matmul(attn_weights, V)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dot_mode=DOT_MODE_STANDARD,
                 softmax_mode=SOFTMAX_MODE_STANDARD):
        super().__init__()
        head_dim = n_embd // n_heads
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn = SelfAttention(n_heads, head_dim, block_size, dot_mode=dot_mode,
            softmax_mode=softmax_mode)
        
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
    def __init__(self, vocab_size, n_embd=64, block_size=64, n_heads=4, n_layers=4, dot_mode=DOT_MODE_STANDARD,
                 softmax_mode=SOFTMAX_MODE_STANDARD):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.block_size = block_size
        
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_heads, block_size, dot_mode, softmax_mode) for _ in range(n_layers)
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