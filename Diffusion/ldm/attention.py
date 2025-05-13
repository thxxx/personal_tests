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
        print("output shape", output.shape)

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
        self.cross_attn = CrossAttention(in_channels, n_heads=8, context_dim=context_dim)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_channels, inner_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(inner_dim, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, context=None):
        x = self.self_attn(self.norm1(x)) + x
        if context:
            x = self.cross_attn(self.norm2(x), context=context) + x
        x = self.feed_forward(self.norm3(x)) + x

        return x