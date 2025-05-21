import torch
import math

def cosine_beta_schedule(timesteps, s = 0.008, device:str = 'cuda'):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0, 0.999)
    
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0).clamp(min=1e-10)

    return betas.to(device), alphas.to(device), alphas_bar.to(device)

def linear_beta_schedule(beta_0=0.0001, beta_T=0.02, timesteps=1000, device:str = 'cuda'):
    betas = torch.linspace(beta_0, beta_T, steps=timesteps, dtype = torch.float32)
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0).clamp(min=1e-10)

    return betas.to(device), alphas.to(device), alphas_bar.to(device)

