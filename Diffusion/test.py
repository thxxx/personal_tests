from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.CenterCrop(178),     # 원본은 (218, 178), 중앙 crop
    transforms.Resize(128),         # 원하는 사이즈로 resize
    transforms.ToTensor(),          # [0, 255] → [0, 1]
    transforms.RandomHorizontalFlip(p=0.1),
])

train_dataset = CelebA(
    root="./data",           # 다운로드 위치
    split="train",           # 또는 "valid", "test"
    download=True,
    transform=transform
)
valid_dataset = CelebA(
    root="./data",
    split="valid",
    download=True,
    transform=transform
)

train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)