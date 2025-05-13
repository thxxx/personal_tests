# cross attention + multi-head attention 구현
# transformer 블럭 구현 - self > cross > FeedForward
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math

class ConvPositionEmbed(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd for ConvPositionEmbed"
        self.conv1 = nn.Conv2d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=16
        )
        self.conv2 = nn.Conv2d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=16
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin = x
        x = self.conv1(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.gelu(x)
        return x + origin # residual connection

class CrossAttention(nn.Module):
    """
    This is used for both self attention and cross attention mechanism
    """
    def __init__(self, in_c, n_heads=8, is_cross=False, context_dim=None):
        super(CrossAttention, self).__init__()
        self.num_heads = n_heads # model_dim을 num_heads개의 head로 나눈 다음 각각 연산하고, 합친다?
        self.is_cross = is_cross
        ctx_dim = context_dim if context_dim is not None else in_c

        self.query = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv2d(ctx_dim, in_c, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(ctx_dim, in_c, kernel_size=1, stride=1, padding=0)

        self.to_out = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None):
        B, C, H, W = x.shape
        # 여기서 d_k = model_dim. 차원이 커질수록 dot product 연산 값 자체도 커진다(weighed sum이기 때문에). 그래서 Attention score가 너무 커지는걸 방지하고자

        # context is used as Key, Value
        if self.is_cross and context is not None:
            K = self.key(context)
            V = self.value(context)
        else:
            K = self.key(x)
            V = self.value(x)
        Q = self.query(x)

        # Q = Q.permute(0, 2, 3, 1).view(B, H*W, C) # rearrange(Q, 'b c h w -> b (h w) c')
        # K = K.permute(0, 2, 3, 1).view(B, H*W, C)
        # V = V.permute(0, 2, 3, 1).view(B, H*W, C)

        Q = rearrange(Q, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        K = rearrange(K, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        V = rearrange(V, 'b (h d) x y -> b h (x y) d', h=self.num_heads)

        # Scaled dot product attention
        # torch.bmm은 batch단위 matmul. 따라서 입력이 3차원 이어야함. 더 큰 경우를 처리하진 못한다.
        # attention_weight = F.softmax(torch.bmm(Q, K)/(int(C//self.num_heads) ** 0.5)) # seq_len 기준으로 softmax 적용.
        # # torch.bmm 이후에 b h (x y) (x y) shape의 attention map이 된다. 각각 (x y)는 key, query의 element들을 의미하고, 서로 얼마나 연관 있는지에 대한 정보가 됨.
        # output = torch.bmm(attention_weight, V) # bs, n_heads, (x y), d
        # # output = output.view(B, H, W, C).permute(0, 3, 1, 2)
        # output = rearrange(output, 'b h (x y) d -> b (h d) x y', x=H, y=W)


        attn_scores = torch.einsum('bhid,bhjd->bhij', Q, K) / (int(C//self.num_heads) ** 0.5)  # [B, heads, HW_q, HW_k]
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.einsum('bhij,bhjd->bhid', attn_weights, V)  # [B, heads, HW, head_dim]

        # Concat heads
        output = rearrange(output, 'b h (x y) d -> b (h d) x y', x=H, y=W)

        output = self.to_out(output)

        return output

class CrossAttention2(nn.Module):
    def __init__(self, in_channels, dim_head=64, n_heads=8, context_dim=None):
        super(CrossAttention2, self).__init__()
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.in_channels = in_channels

        self.context_dim = context_dim if context_dim is not None else in_channels
        inner_dim = dim_head * n_heads

        self.to_q = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.to_k = nn.Conv2d(self.context_dim, inner_dim, kernel_size=1, stride=1, padding=0)
        self.to_v = nn.Conv2d(self.context_dim, inner_dim, kernel_size=1, stride=1, padding=0)

        self.w = nn.Sequential(
            nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Dropout(0.)
        )
        self.scale = dim_head ** 0.5


    def forward(self, x, context=None):
        bs, c, h, w = x.shape
        q = self.to_q(x)
        if context != None:
            context = context.view(bs, self.context_dim, 1, 1).expand(-1, -1, h, w)
        else:
            context = x
        k = self.to_k(context)
        v = self.to_v(context) # bs, dim_head * n_heads, h, w

        q = q.view(bs, self.n_heads, self.dim_head, h*w).permute(0, 1, 3, 2) # h*w는 sequence length, token 수로 해석된다. pixel간의 관계가 중요함.
        k = k.view(bs, self.n_heads, self.dim_head, h*w)
        v = v.view(bs, self.n_heads, self.dim_head, h*w)

        attention_weight = F.softmax(torch.matmul(q, k)/self.scale, dim=-1) # bs, n_heads, h*w, h*w
        output = torch.matmul(v, attention_weight) # bs, n_heads, dim_head, h*w 
        # torch.bmm은 batch단위 matnul. 따라서 입력이 3차원 이어야함

        output = rearrange(output, 'b n d (h w) -> b (n d) h w', h=h, w=w)
        output = self.w(output)

        return output

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

class TransformerBlock(nn.Module):
    """
    Transformer block
    """
    def __init__(self, in_channels, resolution, n_heads=8, context_dim=None, mult=2):
        super(TransformerBlock, self).__init__()
        # self attn
        # cross attn
        # feedforward
        inner_dim = in_channels * mult
        
        self.norm1 = nn.BatchNorm2d(in_channels)
        if context_dim is not None:
            self.norm2 = nn.BatchNorm2d(in_channels)
        self.norm3 = nn.BatchNorm2d(in_channels)

        self.self_attn = CrossAttention(in_channels, n_heads=8)
        if context_dim is not None:
            self.cross_attn = CrossAttention(in_channels, n_heads=8, context_dim=context_dim)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_channels, inner_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(inner_dim, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, t_emb, context=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t_emb.chunk(6, dim=1)

        saout = self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = saout + x * gate_msa.unsqueeze(-1).unsqueeze(-1)
        if context:
            x = self.cross_attn(self.norm2(x), context=context) + x
        x = self.feed_forward(modulate(self.norm3(x), shift_mlp, scale_mlp)) + x * gate_mlp.unsqueeze(-1).unsqueeze(-1)

        return x

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

class DiT(nn.Module):
    def __init__(self, in_dim, model_dim, resolution, depth, num_heads):
        super(DiT, self).__init__()
        self.prev = nn.Conv2d(in_dim, model_dim, kernel_size=1, stride=1, padding=0)
        self.layers = nn.ModuleList(
            TransformerBlock(model_dim, resolution=resolution, n_heads=num_heads, context_dim=None, mult=2)
            for idx in range(depth)
        )
        self.to_out = nn.Conv2d(in_dim, model_dim, kernel_size=1, stride=1, padding=0)
        self.time_embed = TimeEncoding(model_dim)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 6 * model_dim, bias=True)
        )
        self.positional_enc = ConvPositionEmbed(model_dim, kernel_size=3)
    
    def forward(self, x, time):
        # 아마 time의 shape은 (BS,)
        t_emb = self.time_embed(time)
        t_emb = self.time_mlp(t_emb)

        x = self.positional_enc(x)

        x = self.prev(x)
        for layer in self.layers:
            x = layer(x, t_emb=t_emb.squeeze())
        x = self.to_out(x)

        return x
