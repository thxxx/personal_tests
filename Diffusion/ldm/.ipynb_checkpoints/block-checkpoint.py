# 그럼 32*32*3 rgb이미지가 입력으로 들어온다.
# 한 resolution에서 conv(conv-group norm-relu) -> self-attention -> conv
# 32 -> 16 -> 8 -> 4
import torch.nn as nn
import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

class Block(nn.Module):
    """
    conv -> bn -> time embedding add -> activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, is_dropout=False):
        super(Block, self).__init__()
        self.conv=nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.GroupNorm(32, in_channels)
        self.relu = nn.SiLU()
        self.dropout = None
        if is_dropout:
            self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        if self.dropout:
            out = self.dropout(out)

        out = self.conv(out)

        return out

class ResBlock(torch.nn.Module):
    """
    ConvBlock -> attention -> ConvBlock + Residual
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(ResBlock, self).__init__()
        self.block1 = Block(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, is_dropout=False)
        self.block2 = Block(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, is_dropout=True)
        
        self.time_emb_dim = time_emb_dim
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels*2) # 앞에 activation func 한개 있어야 한다 일단 생략
        
        self.residual = nn.Identity()
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, time_emb, context=None):
        time_emb = self.time_mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        scale_shift = time_emb.chunk(2, dim=1)
        scale, shift = scale_shift
        
        out = self.block1(x)
        out = out * (scale+1) + shift
        out = self.block2(out)
        return self.residual(x) + out
