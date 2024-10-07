import torch.nn as nn
import math
import torch

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device=x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        return emb

class DownSample(nn.Module):
    """
    초기에는 Maxpooling도 썼지만 conv stride2가 정보를 더 많이 담을 수 있으니 그걸 쓴다.
    """
    def __init__(self, chn):
        super(DownSample, self).__init__()
        self.down = nn.Conv2d(chn, chn, kernel_size=3, stride=2, padding=1)

    def forward(self, x, t_embed, context=None):
        return self.down(x)

class UpSample(nn.Module):
    def __init__(self, chn, chn_out):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(chn, chn_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t_embed, context=None):
        x = self.up(x)
        x = self.conv(x)
        return x