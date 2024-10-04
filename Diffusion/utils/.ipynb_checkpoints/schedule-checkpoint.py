import torch

def linear_beta_schedule(beta_0=0.0001, beta_T=0.02, timesteps=1000):
    betas = torch.linspace(beta_0, beta_T, steps=timesteps+1)
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    return betas, alphas, alphas_bar

