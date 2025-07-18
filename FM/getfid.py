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
from cfm_0715 import Cfm
import math
import time
import random
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

config = {
    "in_dim":4,
    "model_dim": 768,
    "depth":16,
    "num_heads":8,
    "batch_size": 1024,
    "learning_rate": 1e-4,
    "epochs": 100,
    "sampling_steps": 64,
    "latent_scale": 0.18215,
    "output_dir": "./logs_fm_0718imgnet_fid",
    "beta_1": 0.9,
    "beta_2": 0.99,
    'weight_decay': 0.001,
    "sigma_min": 1e-5,
    "sampling_method": "lognorm",
    "patch_size": 2
}
cfg = SimpleNamespace(**config)
device = 'cuda'

tb_writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, 'runs'))

# downsampling rate=8, latent_dim=4
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
vae.eval()

model = Cfm(
    in_dim=cfg.in_dim, 
    dim=cfg.model_dim, 
    depth=cfg.depth, 
    heads=cfg.num_heads,
    num_classes=1000,
    attn_dropout=0.0,
    ff_dropout=0.1,
    patch_size=cfg.patch_size
)
model.to(device)
print(count_parameters(model))
model = torch.compile(model)

class PrecomputedLatentDataset(Dataset):
    def __init__(self, latent_folder):
        # 저장된 .pt 파일 목록 (정렬 중요)
        self.files = sorted(glob.glob(f"{latent_folder}/*.pt"))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        # key 이름은 train 루프에서 쓰던 것과 맞춰주세요
        return {
            "z_0": data["z"],      # latent 이미지
            "label": data["label"]
        }

with open('labels_for_fid_40960.json', 'r') as f:
    labels = json.load(f)

epoch = 40960

model.eval()
valid_loss = 0

os.makedirs(cfg.output_dir, exist_ok=True)
tqdm_bar = tqdm(total=160, desc="Diffusion FID")

def write(text):
    with open(f'{cfg.output_dir}/logs.txt', 'a') as file:
        file.write(text)


def save_image(index, image_tensor):
    image = image_tensor.squeeze()
    pil_image = TF.to_pil_image(image)
    pil_image.save(os.path.join(save_dir, f"image_{cnt}_{index}.png"))

with torch.no_grad():
    fid_score = None
    
    save_dir = f"{cfg.output_dir}/validsets_fid"
    os.makedirs(save_dir, exist_ok=True)
    bs = 256
    seq_len = 8*8
    for cnt in tqdm(range(160)):
        y0 = torch.randn((bs, 4, 8, 8), device=device)
        t = torch.linspace(0, 1, 64, device=device, dtype=torch.float32)
        context = torch.zeros_like(y0).to(device)
        img_mask = torch.ones((bs, seq_len//4)).to(device).bool()

        label_class = torch.tensor(labels[bs*cnt: bs*cnt + bs]).to(device)

        # ODE function 정의
        class ODEWrapper(nn.Module):
            def __init__(self, model, context, mask, cls):
                super().__init__()
                self.model = model
                self.context = context
                self.mask = mask
                self.cls = cls
            
            def forward(self, t, y):
                return self.model.cfg(
                    w=y,
                    context=self.context,
                    times=t.expand(y.shape[0]),
                    mask=self.mask,
                    cls=self.cls,
                    alpha=3.0
                )
    
        ode_func = torch.compile(ODEWrapper(model, context, img_mask, label_class))
        # ode_func = ODEWrapper(model, context, img_mask, label_class)

        solution = odeint(ode_func, y0, t, method='rk4')
        
        predicted_image = vae.decode(solution[-1] / cfg.latent_scale)['sample']
        predicted_image = (predicted_image.cpu().detach().clamp(0, 1))
        tqdm_bar.update()

        # for i in range(predicted_image.shape[0]-1):
        #     image = predicted_image[i].squeeze()
        #     pil_image = TF.to_pil_image(image)
        #     pil_image.save(os.path.join(save_dir, f"image_{cnt}_{i}.png"))
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            for i in range(predicted_image.shape[0]):
                executor.submit(save_image, i, predicted_image[i])
    
    fid_score = fid.compute_fid(
        "/workspace/personal_tests/valid_set_imagenet/", 
        save_dir
    )
    print(f"\n\nfid score : {fid_score}\n\n")
    write(f'Epoch {epoch} Fid Score - {fid_score}\n\n')





