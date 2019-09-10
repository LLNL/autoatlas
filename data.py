#from scipy import ndimage
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
'''
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
'''

class HCPDataset(Dataset):
    def __init__(self,files,dims=None,mean=None,stdev=None):
        if not isinstance(files,list):
            files = [files]
        self.files = files
        self.dims = dims
        self.mean = mean
        self.stdev = stdev

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        vol = nib.load(self.files[idx]).get_fdata()
        if self.dims is not None:
            #self.dims = np.array(self.dims,dtype=int)
            #iB = (np.array(vol.shape,dtype=int)-self.dims)//2
            #iE = self.dims+iB 
            #vol = vol[iB[0]:iE[0],iB[1]:iE[1],iB[2]:iE[2]]       
            vol = vol[:self.dims[0],:self.dims[1],:self.dims[2]]
        if self.mean is not None:
            vol = vol-self.mean
        if self.stdev is not None:
            vol = vol/self.stdev       
        vol = torch.from_numpy(vol)
        return torch.unsqueeze(vol,dim=0).float()
        
