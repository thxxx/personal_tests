from .block import ResBlock
from .module import UpSample, DownSample, SinusoidalPositionalEmbedding
import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from .attention import TransformerBlock

class Unet(nn.Module):
    def __init__(
        self,
        mults=[1, 2, 4, 4],
        init_resolution=32,
        in_channels=4,
        model_channels=256,
        num_res_blocks=2,
        attention_resolutions=[0, 1, 2],
        context_dim=512,
    ):
        super(Unet, self).__init__()
        
        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, stride=1, padding=1)
        self.last_conv = nn.Conv2d(model_channels, in_channels, kernel_size=3, stride=1, padding=1)
        
        time_dim = model_channels*4
        sinu_pos_embedding = SinusoidalPositionalEmbedding(model_channels, 10000)
        
        self.time_mlp = nn.Sequential(
            sinu_pos_embedding,
            nn.Linear(model_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        chn = model_channels
        resolution = init_resolution
        self.downblocks = nn.ModuleList()
        for level, mult in enumerate(mults):
            out_channels = model_channels * mult # it's not cumulative
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(in_channels, out_channels, time_emb_dim=time_dim)
                )
                if level in attention_resolutions:
                    self.downblocks.append(
                        TransformerBlock(out_channels, resolution=resolution, n_heads=8, context_dim=context_dim, mult=2)
                    )
                in_channels = out_channels
            if level != len(mults)-1:
                self.downblocks.append(
                    DownSample(out_channels)
                )
                resolution /= 2 # Downsampled
        
        middle_channel = out_channels*2
        self.middleblocks = nn.ModuleList([
            ResBlock(out_channels, middle_channel, time_emb_dim=time_dim),
            TransformerBlock(middle_channel, resolution=resolution, n_heads=8, context_dim=context_dim, mult=2),
            ResBlock(middle_channel, middle_channel, time_emb_dim=time_dim)
        ])

        in_channels=middle_channel
        self.upblocks = nn.ModuleList()
        for level, mult in enumerate(mults[::-1]):
            out_channels = model_channels*mult
            for _ in range(num_res_blocks):
                self.upblocks.append(
                    ResBlock(out_channels*2, out_channels, time_emb_dim=time_dim)
                )
                if len(mults) - level - 1 in attention_resolutions:
                    self.upblocks.append(
                        TransformerBlock(out_channels, resolution=resolution, n_heads=8, context_dim=context_dim, mult=2)
                    )
                in_channels = out_channels
            if level != len(mults)-1:
                self.upblocks.append(
                    UpSample(out_channels)
                )
                resolution *= 2

    def forward(self, x, t, context=None):
        initial = self.init_conv(x)
        t_emb = self.time_mlp(t) # 같은 t_emb가 각 ResNet Block에 들어간다.
        
        x = self.init_conv(x)

        connections = []
        for i, layer in enumerate(self.downblocks):
            x = layer(x, t_emb)
            if i%2==0:
                connections.append(x)
        
        for layer in self.middleblocks:
            x = layer(x, t_emb)

        for i in range(len(self.upblocks)):
            x = self.upsamples[i](x, t_emb)
            x = torch.concat((x, connections[::-1][i]), dim=1)
            x = self.upblocks[i](x, t_emb)

        x = self.last_conv(x)
        
        return x