from scipy import ndimage
import nibabel as nib
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SimDataset(Dataset):
    def __init__(self,files):
        if not isinstance(files,list):
            files = [files]
        self.files = files
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        with h5py.File(self.files[idx],'r') as f:
            data = torch.from_numpy(np.array(f['phantom']))
        return torch.unsqueeze(data,dim=0)

class SSDataset(Dataset):
    def __init__(self,files,dims=None):
        if not isinstance(files,list):
            files = [files]
        self.files = files
        self.dims = dims

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        vol = nib.load(self.files[idx]).get_fdata()
        if self.dims is not None:
            factor = np.array(self.dims)/vol.shape
            vol = ndimage.zoom(vol,factor)
        vol = torch.from_numpy(vol)
        return torch.unsqueeze(vol,dim=0).float()
