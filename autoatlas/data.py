import nibabel as nib
import h5py
import numpy as np
from skimage.util import montage
from skimage.filters import threshold_otsu
import os
import torch
from torch.utils.data import Dataset

class NibData(Dataset):
    def __init__(self,data_files,mask_files):
        if not isinstance(data_files,list):
            data_files = [data_files]
        if not isinstance(mask_files,list):
            mask_files = [mask_files]
        assert len(mask_files)==len(data_files)
        self.data_files = data_files
        self.mask_files = mask_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self,idx):
        data_filen = self.data_files[idx]
        data_vol = nib.load(data_filen).get_fdata()
        mask_filen = self.mask_files[idx]
        mask_vol = nib.load(mask_filen).get_fdata()
   
        assert data_vol.shape==mask_vol.shape
        assert data_vol.ndim==3 or data_vol.ndim==2
 
        data_vol = torch.unsqueeze(torch.from_numpy(data_vol),dim=0)
        mask_vol = torch.unsqueeze(torch.from_numpy(mask_vol),dim=0)
        return data_vol.float(),mask_vol.float(),data_filen,mask_filen
#TEMPORARY: SHOULD INCLUDE CODE TO CHECK SAME METADATA FOR MASK AND DATA FILES IN DATA READER

