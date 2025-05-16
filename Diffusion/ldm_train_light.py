# celelb에서 이미지 읽어오는 Dataloader 만들기
# vae latent shape에 대응하도록 모델 짜기
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
from diffusers import AutoencoderKL
from model.unet import Unet
import os
from utils.utils import visualize
from utils.sample import ddim_sample, x_t_sample
import torch.nn.functional as F
from dataset import CelebDataset
from torchvision import transforms
from utils.schedule import linear_beta_schedule
from ldm.attention import DiT
from einops import rearrange

cfg = {
    'in_dim':8,
    "model_dim": 256,
    "depth":16,
    "num_heads":8,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "epochs": 1000,
    "total_timesteps": 1000,
    "beta_0": 0.0001,
    "beta_T": 0.02,
    "sampling_steps": 1000,
    "latent_scale": 0.18215,
    "output_dir": "./logs_dit",
    "beta_1": 0.9,
    "beta_2": 0.95,
    'weight_decay': 0.1
}

def __main__(cfg):
    device = 'cuda'
    
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)

    model = DiT(
        in_dim=cfg.in_dim, 
        model_dim=cfg.model_dim, 
        depth=cfg.depth, 
        num_heads=cfg.num_heads
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/valid_imgs/", exist_ok=True)
    os.makedirs(f"{cfg.output_dir}/weights/", exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta_1, cfg.beta_2), weight_decay=cfg.weight_decay)

    trainer = {
        'train_losses': [],
        'valid_losses': [],
        'valid_images': [],
    }

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.Normalize([0.5], [0.5])
    ])
    
    celeb_dataset = CelebDataset("./celeb", transforms=transform, cache_path="./cache.pt")
    
    train_size = int(0.9 * len(celeb_dataset))
    valid_size = len(celeb_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(celeb_dataset, [train_size, valid_size])
    
    celeb_dataloader_train = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=4
    )
    celeb_dataloader_valid = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False,
        num_workers=4
    )

    betas, alphas, alphas_bar = linear_beta_schedule(cfg.total_timesteps)
    def write(text):
        with open(f'{output_dir}/logs.txt', 'a') as file:
            file.write(text)
    
    from datetime import datetime
    write(f"\n\nTraining start : {datetime.today().strftime('%Y-%m-%d %H:%M')}\n\n")
    
    torch.cuda.empty_cache()
    
    vae.eval()
    last_saved_epoch = 0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0
        tqdm_bar = tqdm(total=len(celeb_dataloader_train), desc="Diffusion Training")
        
        for idx, data in enumerate(celeb_dataloader_train):
            optimizer.zero_grad()
            
            x_0 = data.to(device)
            b, c, h, w = x_0.shape
            with torch.no_grad():
                z_0 = vae.encode(x_0)
                z_0 = z_0['latent_dist'].sample() * cfg.latent_scale
            
            timesteps = torch.randint(1, 1001, (b,))
            added_noise = torch.randn_like(z_0)
            
            # make noised latent by using DDPM form
            z_t = x_t_sample(z_0, timesteps, added_noise)
            z_t = z_t.to(device)
            timesteps = timesteps.to(device)
            added_noise = added_noise.to(device)

            # predict noise
            predicted_noise = model(z_t, timesteps)
            
            loss = F.mse_loss(added_noise, predicted_noise)
            
            loss.backward()
            optimizer.step()
            
            tqdm_bar.update()
            epoch_loss += loss.cpu().detach().item()
            if idx%100==99 and epoch>0:
                trainer['train_losses'].append(epoch_loss/idx)
            
        train_text=f'Epoch {epoch} Train loss - {epoch_loss/len(celeb_dataloader_train)}\n'
        write(train_text)
        
        del loss
        del predicted_noise
        del added_noise
        del timesteps
        del z_t
        torch.cuda.empty_cache()
    
        model.eval()
        valid_loss = 0
        tqdm_bar = tqdm(total=len(celeb_dataloader_valid), desc="Diffusion validation")
        with torch.no_grad():
            for idx, data in enumerate(celeb_dataloader_valid):
                x_0 = data.to(device)
                b, c, h, w = x_0.shape
                with torch.no_grad():
                    z_0 = vae.encode(x_0)
                    z_0 = z_0['latent_dist'].sample() * cfg.latent_scale
                
                timesteps = torch.randint(1, 1001, (b,))
                added_noise = torch.randn_like(z_0)
                
                z_t = x_t_sample(z_0, timesteps, added_noise)
                z_t = z_t.to(device)
                timesteps = timesteps.to(device)
                added_noise = added_noise.to(device)
        
                predicted_noise = model(z_t, timesteps)
                
                loss = F.mse_loss(added_noise, predicted_noise)
                
                valid_loss += loss.cpu().detach().item()
                tqdm_bar.update()

                # Inference Sampling for visual evaluation
                if idx==0:
                    ddim_timesteps = torch.linspace(1000, 10, 100).int().to(device)
                    z_t = torch.randn_like(z_0).to(device) # start from pure Gaussian noise
                    for t in ddim_timesteps:
                        t = rearrange(t, '1 -> bs 1', bs=b) # t = t.repeat(b)
                        predict_noise = model(z_t, t)
                        # denoise xt to x_{t-1}
                        z_t = ddim_sample(z_t, t, predict_noise, alphas_bar[t], alphas_bar[t-10])
    
                    predicted_image = vae.decode(z_t)
                    trainer['valid_images'].append(predicted_image['sample'][:8].cpu().detach()/2+0.5)
                    del ddim_timesteps
                    del predicted_image
        
        del loss
        del predicted_noise
        del added_noise
        del timesteps
        del z_t
        torch.cuda.empty_cache()

        # per data loss
        val_loss = valid_loss/(len(celeb_dataloader_valid) * cfg.batch_size)
        
        trainer['valid_losses'].append(val_loss)
    
        if val_loss <= min(trainer['valid_losses']) or last_saved_epoch+5 < epoch:
            torch.save(model.state_dict(), f'{cfg.output_dir}/weights/model_{epoch}.pth')
            last_saved_epoch = epoch
        
        valid_text=f'Epoch {epoch} Validation loss - {val_loss}\n\n'
        write(valid_text)
    
        plt.plot(trainer['train_losses'])
        plt.savefig(f'{cfg.output_dir}/train_loss.png')
        plt.close()
    
        plt.plot(trainer['valid_losses'])
        plt.savefig(f'{cfg.output_dir}/valid_loss.png')
        plt.close()
    
        visualize(trainer['valid_images'][-1], epoch=epoch, save=True, output_dir=cfg.output_dir)