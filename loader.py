import pandas as pd
import torch
import numpy as np
import os
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

class FashionMNISTDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.csv_path = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.csv_path)

    def __getitem__(self,item):
        img_name = os.path.join(self.data_dir, str(self.csv_path.iloc[item,0]))
        image = io.imread(img_name+".png")
        label = os.path.join(self.data_dir, str(self.csv_path.iloc[item,1]))
        sample = {"image": image, "label":label}

        if self.transform:
            sample = self.transform(image)

        return sample


class FashionMNISTDatasetaftersplit(Dataset):
    def __init__(self, data_dir, X, y, transform=None):
        self.data_dir = data_dir
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self,item):
        img_name = os.path.join(self.data_dir, str(self.X[item]))
        image = io.imread(img_name+".png")
        print(f"Inside get item {image.shape}")
        label = os.path.join(self.data_dir, str(self.y[item]))
        sample = {"image": image, "label":label}

        if self.transform:
            sample = self.transform(image)

        return sample


        
        