from models.lightningdit import LightningDiT_B_2
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from diffusers import AutoencoderKL
from types import SimpleNamespace
from torchdiffeq import odeint_adjoint as odeint # odeint_adjoint는 역전파 효율성을 위해 주로 사용됩니다.
from utils import visualize, show_tensor_image, count_parameters
from cleanfid import fid
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision import transforms
from einops import rearrange
import json
import math
import time
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from torch.optim.lr_scheduler import LambdaLR

def get_custom_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay  # scaled so final LR = base_lr * min_lr_ratio
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

torch.set_float32_matmul_precision('high') # 학습 효율(성능 변화 X)

config = {
    "in_dim":4,
    "latent_res": 8,
    "model_dim": 768,
    "depth":12,
    "num_heads":12,
    "batch_size": 1024,
    "learning_rate": 2e-4,
    "epochs": 80,
    "sampling_steps": 64,
    "latent_scale": 0.18215,
    "output_dir": "./logs_lightningDiT_0718_imagenet64_10240",
    "beta_1": 0.9,
    "beta_2": 0.95,
    'weight_decay': 0.0,
    "sigma_min": 1e-5,
    "sampling_method": "lognorm",
    "patch_size": 2
}
cfg = SimpleNamespace(**config)
device = 'cuda'
tb_writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, 'runs'))

import torch._dynamo
torch._dynamo.config.suppress_errors = True

model = LightningDiT_B_2(
    in_channels=cfg.in_dim,
    input_size=cfg.latent_res, # resolution of latent
    num_classes=1000,
    use_qknorm=False,
    use_swiglu=True,
    use_rope=True,
    use_rmsnorm=True,
    wo_shift=False,
    learn_sigma=False,
)

# downsampling rate=8, latent_dim=4
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
vae.eval()

model = model.to(device)
print(count_parameters(model))

os.makedirs(cfg.output_dir, exist_ok=True)
os.makedirs(f"{cfg.output_dir}/valid_imgs/", exist_ok=True)
os.makedirs(f"{cfg.output_dir}/weights/", exist_ok=True)

with open(f"{cfg.output_dir}/model_structure.txt", "a") as f:
    print(model, file=f)

trainer = {
    'train_losses': [],
    'train_times': [],
    'train_losses_per_timesteps': [],
    'valid_losses': [],
    'valid_images': [],
    'fid_scores': [],
}

import glob

# Transform 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5])
])

# HuggingFace Dataset → PyTorch Dataset으로 감싸기
class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'label': label
        }

# 데이터셋 로딩
dataset = load_dataset("benjamin-paine/imagenet-1k-64x64")

# train/val wrapping
train_dataset = HFDatasetWrapper(dataset["train"], transform=transform)
val_dataset = HFDatasetWrapper(dataset["validation"], transform=transform)

# DataLoader 생성
train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)

print("train dataloader len : ", len(train_dataloader))
print("valid dataloader len : ", len(valid_dataloader))

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta_1, cfg.beta_2), weight_decay=cfg.weight_decay)
scheduler = get_custom_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_dataloader),
    num_training_steps=cfg.epochs*len(train_dataloader)
)

data = next(iter(train_dataset))

def write(text):
    with open(f'{cfg.output_dir}/logs.txt', 'a') as file:
        file.write(text)

with open('/workspace/personal_tests/FM/labels_for_fid_10240.json', 'r') as f:
    labels = json.load(f)

def sample_timestep(batch_size, dtype):
    # timestep은 0~1000이 아니라 0~1 사이 실수 값 uniform
    if cfg.sampling_method == 'uniform':
        t = torch.rand((batch_size, ), dtype=dtype, device=device)
    elif cfg.sampling_method == "lognorm":
        if random.random()<0.2:
            t = torch.rand((batch_size, ), dtype=dtype, device=device)
        else:
            tnorm = np.random.normal(loc=0, scale=1.0, size=batch_size)
            t = 1 / (1 + np.exp(-tnorm))
            t = torch.tensor(t, dtype=dtype, device=device)
    return t

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torchdiffeq import odeint_adjoint as odeint

def solve_ode(model, y0, times, cls, method='rk4'):
    class ODEFunc(nn.Module):
        def __init__(self, model, cls):
            super().__init__()
            self.model, self.cls = model, cls
        
        def forward(self, t, y):
            return self.model.forward_with_cfg(x=y, t=t.expand(y.shape[0]), y=self.cls, cfg_scale=3.0)
    
    return odeint(ODEFunc(model, cls), y0, times, method=method)


def valid_step(epoch):
    print("\n\nStart validation\n\n")
    model.eval()
    valid_loss = 0.0
    device = next(model.parameters()).device
    sigma_min, scale = cfg.sigma_min, cfg.latent_scale
    save_path = Path(cfg.output_dir) / f"validsets_{epoch}"  

    with torch.no_grad(), tqdm(valid_dataloader, desc="Diffusion validation") as bar:
        for idx, batch in enumerate(bar):
            label = batch['label'].to(device)
            x0 = batch['image'].to(device)
            with torch.no_grad():
                z0 = vae.encode(x0)
                z0 = z0['latent_dist'].sample() * cfg.latent_scale
            
            b = z0.size(0)

            t = sample_timestep(b, dtype=z0.dtype).view(b, 1, 1, 1)
            zT = torch.randn_like(z0)
            target = z0 - zT
            zt = (1 - t)*zT + t*z0

            pred = model(
                x=zt,
                t=t.squeeze(),
                y=label
            )
            loss = F.mse_loss(pred, target)
            valid_loss += loss.item()
            bar.set_postfix(loss=valid_loss/(idx+1))

            if idx == 0:
                _log_samples(epoch, z0, label)

    _log_metrics_and_save(epoch, valid_loss/len(valid_dataloader))

    if epoch % 5 == 4:
        save_path.mkdir(exist_ok=True)
        sized=128
        print("\nGet FID\n")
        with torch.no_grad():
            for cnt in tqdm(range(10240//sized)):
                y0 = torch.randn((sized, *z0.shape[1:]), device=device)
                times = torch.linspace(0, 1, 64, device=device)
                label_chunk = torch.tensor(labels[sized*cnt:sized*cnt+sized], device=device)
                
                samples = solve_ode(model, y0, times, label_chunk)
                
                imgs = vae.decode(samples[-1]/scale)['sample'].cpu()
                for i, img in enumerate(imgs[:-1]):
                    TF.to_pil_image(img).save(save_path/f"img_{cnt}_{i}.png")
            
            fid_val = fid.compute_fid(
                "/workspace/personal_tests/valid_set_imagenet/", str(save_path)
            )
            _log_fid(epoch, fid_val)

    torch.cuda.empty_cache()
    return valid_loss/len(valid_dataloader)


def _log_samples(epoch, z0, label):
    y0 = torch.randn_like(z0)[:8]
    times = torch.linspace(0, 1, 64, device=y0.device)
    
    sol = solve_ode(model, y0, times, label[:8])
    
    imgs = vae.decode(sol[-1]/cfg.latent_scale)['sample'][:8].cpu()
    tb_writer.add_images("Valid/Samples", imgs, epoch)
    visualize(imgs, epoch=epoch, save=True, output_dir=cfg.output_dir)

def _log_metrics_and_save(epoch, loss_val):
    trainer['valid_losses'].append(loss_val)
    tb_writer.add_scalar("Valid/Loss", loss_val, epoch)
    if epoch%2==1:
        torch.save(model.state_dict(), f"{cfg.output_dir}/weights/model_{epoch}.pth")
    write(f"Epoch {epoch} Validation loss - {loss_val}\n\n")
    plt.plot(trainer['valid_losses']); plt.savefig(f"{cfg.output_dir}/valid_loss.png"); plt.close()

def _log_fid(epoch, fid_val):
    trainer['fid_scores'].append(fid_val)
    tb_writer.add_scalar("Valid/FID", fid_val, epoch)
    write(f"Epoch {epoch} Fid Score - {fid_val}\n\n")
    plt.plot(trainer['fid_scores']); plt.savefig(f"{cfg.output_dir}/fid_scores.png"); plt.close()

global_step=0
for epoch in range(cfg.epochs):
    torch.cuda.empty_cache()
    model.train()
    epoch_loss = 0
    tqdm_bar = tqdm(total=len(train_dataloader), desc="Latent DiT DDPM Training")
    
    start_at = time.time()
    for idx, data in enumerate(train_dataloader):
        x_0 = data['image'].to(device)
        label_class = data['label'].to(device)
        
        with torch.no_grad():
            z_0 = vae.encode(x_0)
            z_0 = z_0['latent_dist'].sample() * cfg.latent_scale

        b = z_0.size(0)
        
        eps = torch.randn_like(z_0, device=z_0.device, dtype=z_0.dtype)
        
        t = sample_timestep(b, z_0.dtype).view(b, 1, 1, 1)
        z_t = (1 - t)*eps + t*z_0
        target_vf = z_0 - eps

        predicted_vf = model(
            x=z_t,
            t=t.squeeze(),
            y=label_class,
        )

        loss = torch.nn.functional.mse_loss(target_vf, predicted_vf)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        tqdm_bar.update()
        tqdm_bar.set_postfix(loss=loss.item())
        epoch_loss += loss.cpu().detach().item()
        global_step += 1
        tb_writer.add_scalar("Train/Loss", epoch_loss/global_step, global_step)
        
        for param_group in optimizer.param_groups:
            tb_writer.add_scalar("Train/LR", param_group["lr"], global_step)

    trainer['train_losses'].append(epoch_loss/len(train_dataloader))
    write(f'Epoch {epoch} Train loss - {epoch_loss / len(train_dataloader)}\n')
    tb_writer.add_scalar("Train/EpochLoss", epoch_loss / len(train_dataloader), epoch)
    
    plt.plot(trainer['train_losses'])
    plt.savefig(f'{cfg.output_dir}/train_loss.png')
    plt.close()

    torch.cuda.empty_cache()
    valid_step(epoch)

    print("Epoch end")



