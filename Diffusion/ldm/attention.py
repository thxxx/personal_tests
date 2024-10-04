# cross attention + multi-head attention 구현
# transformer 블럭 구현 - self > cross > FeedForward
import torch.nn as nn
import torch

class CrossAttention(nn.Module):
    