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
from cfm_0707 import Cfm
import math
import time
import random
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

config = {
    "in_dim":4,
    "model_dim": 768,
    "depth":16,
    "num_heads":8,
    "batch_size": 512,
    "learning_rate": 1e-4,
    "epochs": 5,
    "sampling_steps": 64,
    "latent_scale": 0.18215,
    "output_dir": "./logs_fm_0708_imagenet642",
    "beta_1": 0.9,
    "beta_2": 0.99,
    'weight_decay': 0.001,
    "sigma_min": 1e-5,
    "sampling_method": "lognorm"
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
    ff_dropout=0.1
)
model.to(device)
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
# show_tensor_image(next(iter(valid_dataloader))['image'][0])

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta_1, cfg.beta_2), weight_decay=cfg.weight_decay)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_dataloader),
    num_training_steps=cfg.epochs*len(train_dataloader)
)

def write(text):
    with open(f'{cfg.output_dir}/logs.txt', 'a') as file:
        file.write(text)

last_saved_epoch = 0
with open('labels_for_fid.json', 'r') as f:
    labels = json.load(f)

def valid_step(epoch):
    model.eval()
    valid_loss = 0
    tqdm_bar = tqdm(total=len(valid_dataloader), desc="Diffusion validation")
    with torch.no_grad():
        for idx, data in enumerate(valid_dataloader):
            x_0 = data['image'].to(device)
            label_class = data['label'].to(device)
            
            with torch.no_grad():
                z_0 = vae.encode(x_0)
                z_0 = z_0['latent_dist'].sample() * cfg.latent_scale
            
            b, c, h, w = z_0.shape
            z_0 = z_0.permute(0, 2, 3, 1).reshape(b, h*w, c)
            z_T = torch.randn_like(z_0, device=z_0.device, dtype=z_0.dtype)

            t = torch.rand((b, ), dtype=z_0.dtype, device=z_0.device)
            t = rearrange(t, "b -> b () ()")
            
            target_vf = z_0 - (1 - cfg.sigma_min) * z_T
            z_t = (1 - (1 - cfg.sigma_min)*t) * z_T + t * z_0
            
            context = torch.zeros_like(z_0).to(device)
            img_mask = torch.ones((b, seq_len)).to(device).bool()
            
            # 4) 모델 예측
            predicted_vf = model(
                w=z_t,
                context=context,
                mask=img_mask,
                times=t.squeeze(),
                cls=label_class
            )

            # 여기선 어짜피 context가 0이니까 그냥 계산하자
            loss = torch.nn.functional.mse_loss(target_vf, predicted_vf)
            
            valid_loss += loss.cpu().detach().item()
            tqdm_bar.update(1)

            # Inference Sampling for visual evaluation
            if idx==0:
                y0 = torch.randn_like(z_0, device=z_0.device)[:8, ]
                t = torch.linspace(0, 1, 64, device=z_0.device, dtype=z_0.dtype)
                context = torch.zeros_like(y0).to(device)[:8, ]
                img_mask = torch.ones((8, seq_len)).to(device).bool()

                # ODE function 정의
                class ODEWrapper(nn.Module):
                    def __init__(self, model, context, mask, cls):
                        super().__init__()
                        self.model = model
                        self.context = context
                        self.mask = mask
                        self.cls = cls
            
                    def forward(self, t, y):
                        return self.model(
                            w=y,
                            context=self.context,
                            times=t.expand(y.shape[0]),
                            mask=self.mask,
                            cls=self.cls
                        )
            
                ode_func = ODEWrapper(model, context, img_mask, label_class[:8,])

                print("Start")
                # ODE solve
                with torch.no_grad():
                    solution = odeint(ode_func, y0, t, method='rk4')  # shape: [64, b, seq_len, c]
                
                solution = rearrange(solution[-1], "b (x y) c -> b c x y", x=w, y=h)
                predicted_image = vae.decode(solution / cfg.latent_scale)['sample']
                predicted_image = (predicted_image[:8].cpu().detach()).clamp(0, 1)
                
                # -1 ~ 1로 나오니까 0~1로 만들기
                trainer['valid_images'].append(predicted_image)
                tb_writer.add_images("Valid/Samples", predicted_image, epoch)

        fid_score = None
        # Get FID per 3epochs
        if epoch % 1 == 0:
            save_dir = f"{cfg.output_dir}/validsets_{epoch}"
            os.makedirs(save_dir, exist_ok=True)
            for cnt in tqdm(range(8)):
                y0 = torch.randn((256, 64, 4), device=device)
                t = torch.linspace(0, 1, 64, device=device, dtype=torch.float32)
                context = torch.zeros_like(y0).to(device)
                img_mask = torch.ones((256, seq_len)).to(device).bool()

                label_class = torch.tensor(labels[256*cnt: 256*cnt + 256]).to(device)

                # ODE function 정의
                class ODEWrapper(nn.Module):
                    def __init__(self, model, context, mask, cls):
                        super().__init__()
                        self.model = model
                        self.context = context
                        self.mask = mask
                        self.cls = cls
            
                    def forward(self, t, y):
                        return self.model(
                            w=y,
                            context=self.context,
                            times=t.expand(y.shape[0]),
                            mask=self.mask,
                            cls=self.cls
                        )
            
                ode_func = ODEWrapper(model, context, img_mask, label_class)

                print("Fid Make")
                # ODE solve
                with torch.no_grad():
                    solution = odeint(ode_func, y0, t, method='rk4')  # shape: [64, b, seq_len, c]
                
                solution = rearrange(solution[-1], "b (x y) c -> b c x y", x=w, y=h)
                predicted_image = vae.decode(solution / cfg.latent_scale)['sample']
                predicted_image = (predicted_image.cpu().detach()).clamp(0, 1) # bs, 3, 64, 64

                for i in range(predicted_image.shape[0]-1):
                    image = predicted_image[i].squeeze()
                    # image = image.permute(1,2,0)
                    pil_image = TF.to_pil_image(image)  # to_pil_image expects [C,H,W]
                    pil_image.save(os.path.join(save_dir, f"image_{cnt}_{i}.png"))
            # get FID
            fid_score = fid.compute_fid(
                "/workspace/personal_tests/valid_set/", 
                save_dir
            )
            trainer['fid_scores'].append(fid_score)
            tb_writer.add_scalar("Valid/FID", fid_score, epoch)
    
    torch.cuda.empty_cache()
    
    # per data loss
    val_loss = valid_loss/len(valid_dataloader)
    trainer['valid_losses'].append(val_loss)
    tb_writer.add_scalar("Valid/Loss", val_loss, epoch)
    
    # if val_loss <= min(trainer['valid_losses']):
    torch.save(model.state_dict(), f'{cfg.output_dir}/weights/model_{epoch}.pth')
    
    # validation logging
    write(f'Epoch {epoch} Validation loss - {val_loss}\n\n')
    plt.plot(trainer['valid_losses'])
    plt.savefig(f'{cfg.output_dir}/valid_loss.png')
    plt.close()
    
    # fid logging
    if fid_score is not None:
        write(f'Epoch {epoch} Fid Score - {fid_score}\n\n')
        plt.plot(trainer['fid_scores'])
        plt.savefig(f'{cfg.output_dir}/fid_scores.png')
        plt.close()
    
    visualize(trainer['valid_images'][-1], epoch=epoch, save=True, output_dir=cfg.output_dir)

vae.eval()
last_saved_epoch = 0
with open(f'{cfg.output_dir}/configs.json', "w") as f:
    json.dump(config, f)

global_step = 0
for epoch in range(cfg.epochs):
    model.train()
    epoch_loss = 0
    train_losses_per_timesteps = [0]*10
    train_losses_per_timesteps_count = [0]*10
    tqdm_bar = tqdm(total=len(train_dataloader), desc="Latent DiT DDPM Training")
    
    start_at = time.time()
    for idx, data in enumerate(train_dataloader):
        x_0 = data['image'].to(device)
        label_class = data['label'].to(device)

        # # 2) VAE encode
        with torch.no_grad():
            z_0 = vae.encode(x_0)
            z_0 = z_0['latent_dist'].sample() * cfg.latent_scale

        # Flatten latent
        b, c, h, w = z_0.shape
        z_0 = z_0.permute(0, 2, 3, 1).reshape(b, h*w, c)
        z_T = torch.randn_like(z_0, device=z_0.device, dtype=z_0.dtype)

        # timestep은 0~1000이 아니라 0~1 사이 실수 값 uniform
        if cfg.sampling_method == 'uniform':
            t = torch.rand((b, ), dtype=z_0.dtype, device=device)
        elif cfg.sampling_method == "lognorm":
            if random.random()<0.2:
                t = torch.rand((b, ), dtype=z_0.dtype, device=device)
            else:
                tnorm = np.random.normal(loc=0, scale=1.0, size=b)
                t = 1 / (1 + np.exp(-tnorm))
                t = torch.tensor(t, dtype=z_0.dtype, device=device)
        
        t = rearrange(t, "b -> b () ()")
        z_t = (1 - (1 - cfg.sigma_min)*t) * z_T + t * z_0
        target_vf = z_0 - (1 - cfg.sigma_min) * z_T
        
        block_size = 4
        seq_len = h * w
        num_blocks = seq_len // block_size

        mask_ratio = random.uniform(0.7, 1.0)
        # block 단위의 마스크 생성: shape (b, num_blocks)
        block_mask = (torch.rand((b, num_blocks), device=z_0.device) > mask_ratio).float()  # 1: keep, 0: mask
        # block mask를 (b, seq_len)로 확장
        mask = block_mask.repeat_interleave(block_size, dim=1)
        
        # 나머지 짜투리 부분 처리 (예: seq_len이 block_size의 배수가 아닌 경우)
        if mask.shape[1] < seq_len:
            pad = torch.ones((b, seq_len - mask.shape[1]), device=mask.device)
            mask = torch.cat([mask, pad], dim=1)
        
        context = z_0 * mask.unsqueeze(-1)  # (b, h*w, c)
        img_mask = torch.ones((b, seq_len)).to(device).bool()

        # 4) 모델 예측
        predicted_vf = model(
            w=z_t,
            context=context,
            mask=img_mask,
            times=t.squeeze(),
            cls=label_class
        )

        # Loss 계산
        # loss = torch.nn.functional.mse_loss(target_vf, predicted_vf, reduction='none').mean(dim=(1, 2))
        # Loss 계산: 마스킹된 위치 (mask == 0)에서만
        loss_mask = (mask == 0).unsqueeze(-1)  # (b, h*w, 1)
        loss = torch.nn.functional.mse_loss(target_vf, predicted_vf, reduction='none')  # (b, h*w, c)
        masked_loss = loss * loss_mask
        loss = masked_loss.sum(dim=(1, 2)) / (loss_mask.sum(dim=(1, 2)) + 1e-8)
        
        if idx>5:
            for idxin, ptl in enumerate(loss):
                train_losses_per_timesteps[min(math.floor(t[idxin]*10), 9)] += ptl.cpu().detach().item()
                train_losses_per_timesteps_count[min(math.floor(t[idxin]*10), 9)] += 1
        
        loss = loss.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        tqdm_bar.update()
        tqdm_bar.set_postfix(loss=loss.item())
        epoch_loss += loss.cpu().detach().item()

        tb_writer.add_scalar("Train/StepLoss", loss.item(), global_step)
        global_step += 1

        if idx%100==99:
            print("loss - ", loss)

    tb_writer.add_scalar("Train/EpochLoss", epoch_loss/len(train_dataloader), epoch)
    
    trainer['train_times'].append(time.time() - start_at)

    trainer['train_losses'].append(epoch_loss/len(train_dataloader))
    trainer['train_losses_per_timesteps'].append(train_losses_per_timesteps)
    train_text = f'Epoch {epoch} Train loss - {epoch_loss / len(train_dataloader)}\n'
    for i in range(10):
        count = train_losses_per_timesteps_count[i]
        avg = train_losses_per_timesteps[i] / count if count > 0 else 0
        train_text += f"timesteps {i/10:.1f} ~ {i/10+0.1:.1f} : {avg:.4f}\n"
    
    write(train_text)
    
    plt.plot(trainer['train_losses'])
    plt.savefig(f'{cfg.output_dir}/train_loss.png')
    plt.close()

    torch.cuda.empty_cache()
    valid_step(epoch)

    print("Epoch end")

tb_writer.close()