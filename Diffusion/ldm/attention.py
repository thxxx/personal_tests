# cross attention + multi-head attention 구현
# transformer 블럭 구현 - self > cross > FeedForward
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange

class CrossAttention(nn.Module):
    """
    This is used for both self attention and cross attention mechanism
    """
    def __init__(self, in_c, num_heads=8, is_cross=False, context_dim=None):
        self.num_heads = num_heads
        self.is_cross = is_cross
        self.context_dim = context_dim

        self.query = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context):
        B, C, H, W = x.shape
        if self.is_cross and context:
            Q = self.query(context)
        else:
            Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.permute(0, 2, 3, 1).view(B, H*W, C)
        K = K.view(B, C, H*W)
        V = V.permute(0, 2, 3, 1).view(B, H*W, C)

        attention_weight = F.softmax(torch.bmm(Q, K)/(int(C) ** 0.5))
        output = torch.bmm(attention_weight, V)
        output = output.view(B, H, W, C).permute(0, 3, 1, 2)

        return output

class CrossAttention2(nn.Module):
    def __init__(self, in_channels, dim_head=64, n_heads=8, context_dim=None):
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


    def forward(self, x, context):
        bs, c, h, w = x.shape
        q = self.to_q(x)
        context = context if context else x
        k = self.to_k(context)
        v = self.to_v(context) # bs, dim_head * n_heads, h, w

        q = q.view(bs, self.n_heads, self.dim_head, h*w).permute(0, 1, 3, 2) # h*w는 sequence length, token 수로 해석된다.
        k = k.view(bs, self.n_heads, self.dim_head, h*w)
        v = v.view(bs, self.n_heads, self.dim_head, h*w)

        attention_weight = F.softmax(torch.matmul(q, k)/self.scale, dim=-1) # bs, n_heads, h*w, h*w
        output = torch.matmul(v, attention_weight) # bs, n_heads, dim_head, h*w 
        # torch.bmm은 batch단위 matnul. 따라서 입력이 3차원 이어야함

        output = rearrange(output, 'b n d (h w) -> b (n d) h w', h=h, w=w)
        output = self.w(output)

        return output


class TransformerBlock(nn.Module):
    """
    Transformer block
    """
    def __init__(self, in_channels, n_heads=8, context_dim=None):
        # self attn
        # cross attn
        # feedforward
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.norm3 = nn.LayerNorm(in_channels)
        self.self_attn = CrossAttention(in_channels, num_heads=8, is_cross=False)
        self.cross_attn = CrossAttention(in_channels, num_heads=8, is_cross=True, context_dim=context_dim)
        self.feed_forward = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, context):
        x = self.self_attn(self.norm1(x)) + x
        x = self.cross_attn(self.norm2(x), context=context) + x
        x = self.feed_forward(self.norm3(x)) + x

        return x

# class SpatialTransformer(nn.Module):
#     def __init__(self, in_channels, n_heads, context_dim):
#         self.

#     def forward(self, x, context):
#         return x