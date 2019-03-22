import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class SimDataset(Dataset):
    def __init__(self,data_files):
        if not isinstance(data_files,list):
            data_files = [data_files]
        self.files = data_files
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        with h5py.File(self.files[idx],'r') as f:
            data = torch.from_numpy(np.array(f['phantom']))
        return torch.unsqueeze(data,dim=0)
