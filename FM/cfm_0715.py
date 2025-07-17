# This is Dit model with better implementation

import torch
import math
from typing import cast
from einops import einsum, rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

AudioTensor = Float[Tensor, "batch audio audio_channel"]
AudioMaskTensor = Bool[Tensor, "batch audio"]
EncTensor = Float[Tensor, "batch codec channel"]
EncMaskTensor = Bool[Tensor, "batch codec"]
LengthTensor = Int[Tensor, "batch"]
LossTensor = Float[Tensor, ""]
TimeTensor = Float[Tensor, "batch"]
Batch = tuple[AudioTensor, AudioMaskTensor]

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (batch, ..., dim)
        norm = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.scale * x / (norm + self.eps)



class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        slope = -torch.arange(1, self.heads + 1) * 8 / self.heads
        slope = slope.exp2()
        self.register_buffer("slope", slope, persistent=False)

    def get_alibi_bias(self, length: int, device: torch.device):
        arange = torch.arange(length, device=device, dtype=self.slope.dtype)
        rel = rearrange(arange, "q -> q ()") - rearrange(arange, "k -> () k")
        bias = einsum(rel, self.slope, "q k, h -> h q k")
        return -bias.abs()

    def forward(self, x: Tensor, key_mask: EncMaskTensor):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "b q (h d) -> b h q d", h=self.heads)
        k = rearrange(k, "b k (h d) -> b h k d", h=self.heads)
        v = rearrange(v, "b k (h d) -> b h k d", h=self.heads)

        attn_mask = rearrange(key_mask, "b k -> b () () k")
        attn_mask = torch.where(attn_mask, 0, -torch.inf)
        alibi_bias = self.get_alibi_bias(q.shape[2], q.device)
        attn_mask = attn_mask + alibi_bias
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = rearrange(out, "b h q d -> b q (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
        skip: bool,
    ):
        super().__init__()
        self.skip = skip
        self.skip_combiner = nn.Linear(dim * 2, dim) if skip else nn.Identity()

        self.attn_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, heads=heads, dropout=attn_dropout)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

        self.ff_norm = RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 4, dim),
        )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=2, in_channels=4, embed_dim=512):
        super().__init__()
        self.patch_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
    ):
        super().__init__()

        self.layers = cast(
            list[TransformerBlock],
            nn.ModuleList(
                TransformerBlock(
                    dim,
                    heads=heads,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    skip=ind + 1 > (depth // 2),
                )
                for ind in range(depth)
            ),
        )

        self.final_norm = RMSNorm(dim)

    def forward(
        self, x: Tensor, mask: EncMaskTensor, cond: Tensor
    ) -> Tensor:
        
        skip_connects = []
        for layer in self.layers:
            time_emb = layer.time_mlp(cond)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = time_emb.chunk(6, dim=1)

            if layer.skip:
                skip_connect = skip_connects.pop()
                x = torch.cat((x, skip_connect), dim=-1)
                x = layer.skip_combiner(x)
            else:
                skip_connects.append(x)

            attn_input = layer.attn_norm(x)
            attn_input = modulate(attn_input, shift_msa, scale_msa)
            x = layer.attn(attn_input, key_mask=mask) + x * (1 + gate_msa.unsqueeze(1))

            ff_input = layer.ff_norm(x)
            ff_input = modulate(ff_input, shift_mlp, scale_mlp)
            x = layer.ff(ff_input) + x * (1 + gate_mlp.unsqueeze(1))

        return self.final_norm(x)


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

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        origin = x
        mask = rearrange(~mask, "b n -> b () n")
        x = rearrange(x, "b n c -> b c n")
        x = x.masked_fill(mask, 0.0)
        x = self.conv1(x)
        x = self.gelu(x)
        x = x.masked_fill(mask, 0.0)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x.masked_fill(mask, 0.0)
        x = rearrange(x, "b c n -> b n c")
        return x + origin


class TimeEncoding(nn.Module):
    """used by @crowsonkb"""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dimension must be divisible by 2"
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        freqs = 2 * math.pi * einsum(x, self.weights, "b, d -> b d")
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return rearrange(fouriered, "b d -> b () d")


class Cfm(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        depth: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
        num_classes: int,
        patch_size: int = 2
    ):
        super().__init__()

        self.patchify = PatchEmbed(patch_size=patch_size, in_channels=in_dim, embed_dim=dim)
        self.unconditional_cls_idx = num_classes + 1

        # self.combine = nn.Linear(in_dim * 2, dim)
        self.conv_embed = ConvPositionEmbed(dim=dim, kernel_size=3)
        self.time_emb = TimeEncoding(dim)
        self.class_embed = nn.Embedding(num_classes, dim)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
        )

        self.to_pred = nn.Linear(dim, in_dim * patch_size * patch_size, bias=False)
        self.patch_size = patch_size

    def cfg(
        self,
        w: EncTensor,
        context: EncTensor,
        mask: EncMaskTensor,
        times: TimeTensor,
        cls: EncTensor,
        alpha=3.0,
    ) -> EncTensor:
        b, c, h, wid = w.shape
        
        w = repeat(w, "b ... -> (r b) ...", r=2)
        context = repeat(context, "b ... -> (r b) ...", r=2)
        mask = repeat(mask, "b ... -> (r b) ...", r=2)
        times = repeat(times, "b ... -> (r b) ...", r=2)

        patched = self.patchify(w)
        w = self.conv_embed(patched, mask)
        time_emb = self.time_emb(times)
        
        cond = self.class_embed(cls)
        null_cond = torch.zeros_like(cond)
        class_emb = torch.concat((cond, null_cond), dim=0)
        cond = time_emb.squeeze() + class_emb

        w = self.transformer(w, mask=mask, cond=cond)
        out = self.to_pred(w)
        
        out = rearrange(
            out,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h // self.patch_size,
            w=wid // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )
        
        logits, null_logits = out.chunk(2, dim=0)

        return logits + alpha * (logits - null_logits)
    
    def forward(
        self,
        w: EncTensor,
        context: EncTensor,
        mask: EncMaskTensor,
        times: TimeTensor,
        cls: EncTensor,
        drop_condition
    ) -> EncTensor:
        b, c, h, wid = w.shape
        # embed = torch.cat((w, context), dim=-1)
        # combined = self.combine(embed)
        patched = self.patchify(w)

        w = self.conv_embed(patched, mask)

        # timestep & class condition embedding
        time_emb = self.time_emb(times)
        
        class_emb = self.class_embed(cls) # B, dim
        if drop_condition:
            class_emb = torch.zeros_like(class_emb)

        cond = time_emb.squeeze() + class_emb

        w = self.transformer(w, mask=mask, cond=cond)

        w = self.to_pred(w)
        
        w = rearrange(
            w,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h // self.patch_size,
            w=wid // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )
        
        return w





