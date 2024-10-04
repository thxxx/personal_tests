import torch.nn as nn
import torch
import torch.nn.functional as F

class AttnBlock(nn.Module):
    def __init__(self, in_c):
        super(AttnBlock, self).__init__()
        self.query = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.permute(0, 2, 3, 1).view(B, H*W, C)
        k = k.view(B, C, H*W)
        v = v.permute(0, 2, 3, 1).view(B, H*W, C)

        attn_score = torch.bmm(q, k)
        attn_weight = F.softmax(attn_score/(int(C)**0.5), dim=-1)

        output = torch.bmm(attn_weight, v)

        x = output.view(B, H, W, C).permute(0, 3, 1, 2)
        
        return x