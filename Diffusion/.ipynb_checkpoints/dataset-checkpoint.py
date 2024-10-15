import torchvision
from torchvision import transforms
import torch
import os

class CelebDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transforms=None, cache_path=""):
        super(CelebDataset, self).__init__()
        self.transform = transforms

        if os.path.exists(cache_path):
            self.data_list = torch.load(cache_path)
        else:
            self.data_list = []
            for single in tqdm(os.listdir(data_path)):
                single_path = data_path + '/' + single
                img = torchvision.io.read_image(single_path)
                self.data_list.append(img)
            torch.save(self.data_list, cache_path)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx].float()
        if self.transform:
            item = self.transform(item)
        return item