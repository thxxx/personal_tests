import torchvision
from torchvision import transforms
from transformers import CLIPTokenizer
import json
import torch
from tqdm import tqdm

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, transforms=None, is_valid=False):
        super(CocoDataset, self).__init__()
        self.transform = transforms
        self.tokenizer = tokenizer
        with open(data_path, 'r') as f:
            datas = json.load(f)

        self.data_list = []
        self.text_list = []
        for d in tqdm(datas):
            if not is_valid:
                img_path = '/workspace/train2017/' + d['file_name']
            else:
                img_path = '/workspace/val2017/' + d['file_name']
            self.data_list.append(img_path)
            self.text_list.append(d['caption'])
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = self.data_list[idx]
        caption = self.text_list[idx]
        
        img = torchvision.io.read_image(path).float()
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if self.transform:
            img = self.transform(img)
        
        text_tokenized = self.tokenizer(caption, return_tensors="pt", padding='max_length', max_length=77, truncation=True)
        
        return {
            "image":img,
            "caption":text_tokenized
        }