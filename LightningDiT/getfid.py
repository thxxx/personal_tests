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
import glob
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import torch._dynamo
import glob

torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high') # 학습 효율(성능 변화 X)

num_generations = 4096
bs = 256

config = {
    "in_dim":4,
    "latent_res": 8,
    "batch_size": 1024,
    "learning_rate": 2e-4,
    "epochs": 80,
    "sampling_steps": 64,
    "latent_scale": 0.18215,
    "output_dir": "./logs_lightningDiT_0726_imagenet64_fid",
    "beta_1": 0.9,
    "beta_2": 0.95,
    'weight_decay': 0.0,
    "sigma_min": 1e-5,
    "sampling_method": "lognorm"
}
cfg = SimpleNamespace(**config)
device = 'cuda'

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
try:
    state = torch.load('./imagenet_64_0718_47epoch.pth', map_location='cpu')
    model.load_state_dict(state)
except Exception as e:
    print("Error : ", e)
model.eval()

os.makedirs(cfg.output_dir, exist_ok=True)

with open(f"{cfg.output_dir}/model_structure.txt", "a") as f:
    print(model, file=f)

with open('labels_for_fid_4096.json', 'r') as f:
    labels = json.load(f)

def write(text):
    with open(f'{cfg.output_dir}/logs.txt', 'a') as file:
        file.write(text)

def save_image(index, image_tensor):
    image = image_tensor.squeeze()
    pil_image = TF.to_pil_image(image)
    pil_image.save(os.path.join(save_dir, f"image_{cnt}_{index}.png"))

tqdm_bar = tqdm(total=num_generations//bs, desc="Diffusion FID")
def solve_ode(model, y0, times, cls, method='rk4'):
    class ODEFunc(nn.Module):
        def __init__(self, model, cls):
            super().__init__()
            self.model, self.cls = model, cls
        
        def forward(self, t, y):
            return self.model.forward_with_cfg(
                x=y, 
                t=t.expand(y.shape[0]), 
                y=self.cls, 
                cfg_scale=3.0
            )
    
    return odeint(ODEFunc(model, cls), y0, times, method=method)

with torch.no_grad():
    fid_score = None
    
    save_dir = f"{cfg.output_dir}/validsets_fid"
    os.makedirs(save_dir, exist_ok=True)
    for cnt in tqdm(range(num_generations//bs)):
        y0 = torch.randn((bs, 4, 8, 8), device=device)
        times = torch.linspace(0, 1, 64, device=device)
        label_class = torch.tensor(labels[bs*cnt: bs*cnt + bs]).to(device)
        
        samples = solve_ode(model, y0, times, label_class)
        imgs = vae.decode(samples[-1]/cfg.latent_scale)['sample'].cpu().detach().clamp(0, 1)
        tqdm_bar.update()
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            for i in range(imgs.shape[0]):
                executor.submit(save_image, i, imgs[i])
    
    fid_score = fid.compute_fid(
        "/workspace/personal_tests/valid_set_imagenet64/", 
        save_dir
    )
    print(f"\n\nfid score : {fid_score}\n\n")
    write(f'Epoch Fid Score - {fid_score}\n\n')





