import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np

class SimDataset(Dataset):
    def __init__(self,data_folder,num_samples):
        self.files = [os.path.join(data_folder,f) for f in os.listdir(data_folder) if f[-3:]=='.h5']
        if len(self.files) < num_samples:
            print('WARN: Number of files in {} is less than {}.'.format(data_folder,num_samples))
        else:
            self.files = self.files[:num_samples]
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        with h5py.File(self.files[idx],'r') as f:
            data = torch.from_numpy(np.array(f['phantom']))
        return torch.unsqueeze(data,dim=0)
