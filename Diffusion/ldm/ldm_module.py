import torch.nn as nn
from lightning.pytorch import LightningModule
from attention import DiT

class LDMModule(LightningModule):
    def __init__(
            self, 
            model_dim:int, 
            depth :int, 
            num_heads: int,
            in_dim:int,
        ):
        super().__init__()
        self.model = DiT(
            in_dim=in_dim, 
            model_dim=model_dim, 
            depth=depth, 
            num_heads=num_heads
        )
    
    def 