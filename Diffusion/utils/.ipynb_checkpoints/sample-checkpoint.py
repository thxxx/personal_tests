import torch

def ddim_sample(x_t, timesteps, predicted_noise, alphas_bar_t, alphas_bar_t_1):
    q = torch.stack([alphas_bar_t_1[idx].sqrt() * (x_t[idx] - (1-alphas_bar_t[idx]).sqrt() * predicted_noise[idx])/alphas_bar_t[idx].sqrt() for idx, t in enumerate(timesteps)])
    return torch.stack([q[idx] + (1-alphas_bar_t_1[idx]).sqrt() * predicted_noise[idx] for idx, t in enumerate(timesteps)])

def ddpm_sample(x_t, timesteps, predicted_noise, z, betas, alphas, alphas_bar):
    moved_mean = torch.stack([x_t[idx] - (1-alphas[t])/(torch.sqrt(1-alphas_bar[t])) * predicted_noise[idx] for idx, t in enumerate(timesteps)])
    return torch.stack([1/torch.sqrt(alphas[t]) * moved_mean[idx] + torch.sqrt(betas[t]) * z[idx] for idx, t in enumerate(timesteps)])

def x_t_sample(x_0, timesteps, noise):
    return torch.stack([torch.sqrt(alphas_bar[t])*x_0[idx] + torch.sqrt(1-alphas_bar[t])*noise[idx] for idx, t in enumerate(timesteps)])

def x_t_1_sample(x_t, timesteps, predicted_noise, z):
    moved_mean = torch.stack([x_t[idx] - (1-alphas[t])/(torch.sqrt(1-alphas_bar[t])) * predicted_noise[idx] for idx, t in enumerate(timesteps)])
    return torch.stack([1/torch.sqrt(alphas[t]) * moved_mean[idx] + torch.sqrt(betas[t]) * z[idx] for idx, t in enumerate(timesteps)])
