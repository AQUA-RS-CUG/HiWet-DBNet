import os
from random import sample
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class MSCNN_Dataset(Dataset):
    def __init__(
        self, 
        optical_image_list, 
        sar_image_list, 
        label_list2, 
        label_list3
    ):
        super(MSCNN_Dataset, self).__init__()
        self.optical_image_list = optical_image_list
        self.sar_image_list = sar_image_list
        self.label_list2 = label_list2
        self.label_list3 = label_list3
        
    def __getitem__(self, index):
        optical_image = tiff.imread(self.optical_image_list[index]).astype(np.float32)
        sar_image = tiff.imread(self.sar_image_list[index]).astype(np.float32)
        label2 = self.label_list2[index]
        label3 = self.label_list3[index]
        optical_image = torch.from_numpy(optical_image)
        sar_image = torch.from_numpy(sar_image)
        sample = {
            'optical_image':optical_image,
            'sar_image':sar_image,
            'label2':label2,
            'label3':label3
        }
        return sample
    
    def __len__(self):
        return len(self.optical_image_list)
