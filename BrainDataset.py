import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np



class BrainDataset(Dataset):
    def __init__(self, ad_dir, hc_dir, transform=None):
        self.ad_files = [os.path.join(ad_dir, f) for f in os.listdir(ad_dir) if f.endswith('.nii')]
        self.hc_files = [os.path.join(hc_dir, f) for f in os.listdir(hc_dir) if f.endswith('.nii')]
        self.files = self.ad_files + self.hc_files
        self.labels = [1] * len(self.ad_files) + [0] * len(self.hc_files)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]
        img = nib.load(img_path).get_fdata()
        img = np.expand_dims(img, axis=0)  # 添加通道维度
        img = torch.from_numpy(img).float()

        if self.transform:
            img = self.transform(img)

        return img, label