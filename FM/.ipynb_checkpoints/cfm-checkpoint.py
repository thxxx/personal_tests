# cross attention + multi-head attention 구현
# transformer 블럭 구현 - self > cross > FeedForward
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
from torch import einsum

class ConvPositionEmbed(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd for ConvPositionEmbed"
        self.conv1 = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=16
        )
        self.conv2 = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=16
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin = x
        x = rearrange(x, "bs n d -> bs d n")
        x = self.conv1(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.gelu(x)
        x = rearrange(x, "bs d n -> bs n d")
        return x + origin # add positional embedding at the origin

class Attention(nn.Module):
    """
    This is used for both self attention and cross attention mechanism
    """
    def __init__(self, model_dim, n_heads=8, is_cross=False, context_dim=None):
        super().__init__()
        self.num_heads = n_heads # model_dim을 num_heads개의 head로 나눈 다음 각각 연산하고, 합친다?
        self.is_cross = is_cross
        ctx_dim = context_dim if context_dim is not None else model_dim

        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key = nn.Linear(ctx_dim, model_dim, bias=False)
        self.value = nn.Linear(ctx_dim, model_dim, bias=False)

        self.to_out = nn.Linear(model_dim, model_dim, bias=False)

        slope = (-torch.arange(1, 8 + 1) * 4 / 8).exp2()
        self.register_buffer("slope", slope, persistent=False)
    
    def get_alibi_bias(self, length: int, device: torch.device):
        arange = torch.arange(length, device=device, dtype=self.slope.dtype)
        rel = rearrange(arange, "q -> q ()") - rearrange(arange, "k -> () k")
        bias = einsum("q k, h -> h q k", rel, self.slope)
        return -bias.abs()
    
    def forward(self, x, context=None):
        B, SL, D = x.shape

        # context is used as Key, Value
        if self.is_cross and context is not None:
            K = self.key(context)
            V = self.value(context)
        else:
            K = self.key(x)
            V = self.value(x)
        Q = self.query(x)

        Q = rearrange(Q, 'b n (h d) -> b h n d', h=self.num_heads)
        K = rearrange(K, 'b n (h d) -> b h n d', h=self.num_heads)
        V = rearrange(V, 'b n (h d) -> b h n d', h=self.num_heads)
        
        attn_scores = torch.einsum('bhid,bhjd->bhij', Q, K) / (int(D//self.num_heads) ** 0.5)  # [B, heads, HW_q, HW_k]
        
        if context is None:
            alibi = self.get_alibi_bias(SL, Q.device)
            attn_scores = attn_scores + alibi
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.einsum('bhij,bhjd->bhid', attn_weights, V)  # [B, heads, HW, head_dim]

        # Concat heads
        output = rearrange(output, 'b h n d -> b n (h d)')

        output = self.to_out(output)

        return output

class TimeEncoding(nn.Module):
    """used by @crowsonkb"""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dimension must be divisible by 2"
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = 2 * math.pi * torch.einsum("b, d -> b d", x, self.weights) # x는 스칼라값 list. 여기 각 weight가 곱해져서 embedding이 된다.
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return rearrange(fouriered, "b d -> b () d")

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TransformerBlock(nn.Module):
    """
    Transformer block

    self attn
    cross attn(optional)
    feedforward
    """
    def __init__(self, model_dim, n_heads=8, context_dim=None):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        if context_dim is not None:
            self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

        self.self_attn = Attention(model_dim, n_heads=8)
        if context_dim is not None:
            self.cross_attn = Attention(model_dim, n_heads=8, context_dim=context_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, model_dim*4, bias=False),
            nn.GELU(),
            nn.Linear(model_dim*4, model_dim, bias=False),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 6 * model_dim, bias=True)
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, x, t_emb, context=None):
        t_emb = self.time_mlp(t_emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t_emb.chunk(6, dim=1)

        saout = self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = saout + x * (1+gate_msa.unsqueeze(1))
        if context is not None:
            x = self.cross_attn(self.norm2(x), context=context) + x
        x = self.feed_forward(modulate(self.norm3(x), shift_mlp, scale_mlp)) + x * (1+gate_mlp.unsqueeze(1))

        return x

class DiT(nn.Module):
    def __init__(self, in_dim, model_dim, depth, num_heads):
        super(DiT, self).__init__()
        self.prev = nn.Linear(in_dim, model_dim, bias=True)
        self.layers = nn.ModuleList(
            TransformerBlock(model_dim, n_heads=num_heads, context_dim=None)
            for idx in range(depth)
        )
        self.to_out = nn.Linear(model_dim, in_dim, bias=True)
        
        self.time_embed = TimeEncoding(model_dim)
        # self.time_mlp = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(model_dim, 6 * model_dim, bias=False)
        # )
        # nn.init.zeros_(self.time_mlp[-1].weight)
        self.positional_enc = ConvPositionEmbed(model_dim, kernel_size=3)
        # self.pos_enc = SinusoidalPositionalEmbedding(model_dim, max_len=64)

        assert model_dim % num_heads == 0
    
    def forward(self, time, x):
        if time.dim() == 0:
            time = time.expand(x.shape[0])
        # 아마 time의 shape은 (BS,)
        t_emb = self.time_embed(time)
        # t_emb = self.time_mlp(t_emb)

        x = self.prev(x)
        x = self.positional_enc(x)
        
        for layer in self.layers:
            x = layer(x, t_emb=t_emb.squeeze())
        x = self.to_out(x)

        return x
