from .block import ResConvblock
from .module import UpSample, DownSample, SinusoidalPositionalEmbedding
from .attn import AttnBlock
import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

class Unet(nn.Module):
    def __init__(
        self,
        dim=64,
        mults=[1,2,4],
        channel_scale=3
        # mults=[1,2,4,8]
    ):
        super(Unet, self).__init__()
        
        self.init_conv = nn.Conv2d(channel_scale, dim, kernel_size=3, stride=1, padding=1)
        self.last_conv = nn.Conv2d(dim, channel_scale, kernel_size=3, stride=1, padding=1)
        
        time_dim = dim*4
        sinu_pos_embedding = SinusoidalPositionalEmbedding(dim, 10000)
        self.time_mlp = nn.Sequential(
            sinu_pos_embedding,
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        in_channels=dim
        self.downblocks = nn.ModuleList()
        for i, mult in enumerate(mults):
            out_channels = dim*mult
            self.downblocks.append(
                ResConvblock(in_channels, out_channels, is_attn=False, time_emb_dim=time_dim)
            )
            self.downblocks.append(
                DownSample(out_channels)
            )
            in_channels = dim*mult
        
        middle_channel = out_channels*2
        self.middleblocks = nn.ModuleList([
            ResConvblock(out_channels, middle_channel, is_attn=False, time_emb_dim=time_dim),
            ResConvblock(middle_channel, middle_channel, is_attn=False, time_emb_dim=time_dim)
        ])

        in_channels=middle_channel
        self.upsamples = nn.ModuleList()
        for i, mult in enumerate(mults[::-1]):
            out_channels = dim*mult
            self.upsamples.append(
                UpSample(in_channels, out_channels)
            )
            in_channels = out_channels
        
        self.upblocks = nn.ModuleList()
        for i, mult in enumerate(mults[::-1]):
            out_channels = dim*mult
            self.upblocks.append(
                ResConvblock(out_channels*2, out_channels, is_attn=False, time_emb_dim=time_dim)
            )

    def forward(self, x, t):
        initial = self.init_conv(x)
        t_emb = self.time_mlp(t) # 같은 t_emb가 각 ResNet Block에 들어간다.
        
        x = self.init_conv(x)

        connections = []
        # print("Before getting into the Down Blocks : ", x.shape)
        for i, layer in enumerate(self.downblocks):
            x = layer(x, t_emb)
            if i%2==0:
                connections.append(x)
        # print("After down : ", x.shape, len(connections))
        
        for layer in self.middleblocks:
            x = layer(x, t_emb)
        # print("After middle : ", x.shape)

        for i in range(len(self.upblocks)):
            x = self.upsamples[i](x, t_emb)
            # print(f"{i} - ", x.shape)
            x = torch.concat((x, connections[::-1][i]), dim=1)
            # print(f"{i} - ", x.shape)
            x = self.upblocks[i](x, t_emb)

        x = self.last_conv(x)
        
        return x