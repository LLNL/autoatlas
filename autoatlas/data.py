import nibabel as nib
import h5py
import numpy as np
from skimage.util import montage
from skimage.filters import threshold_otsu
import os
import torch
from torch.utils.data import Dataset

class NibData(Dataset):
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
        filename = self.files[idx]
        dptr = nib.load(filename)
        vol = dptr.get_fdata()
        if self.dims is not None:
            self.dims = np.array(self.dims,dtype=int)
            iB = (np.array(vol.shape,dtype=int)-self.dims)//2
            iE = self.dims+iB
            if iB[0]>=0: 
                vol = vol[iB[0]:iE[0]]
            if iB[1]>=0: 
                vol = vol[:,iB[1]:iE[1]]
            if iB[2]>=0: 
                vol = vol[:,:,iB[2]:iE[2]]
            sh = np.array(vol.shape,dtype=int)
            wid = self.dims-sh
            wid[wid<0] = 0
            vol = np.pad(vol,((0,wid[0]),(0,wid[1]),(0,wid[2])),mode='constant')
        if self.mean is not None:
            vol = vol-self.mean
        if self.stdev is not None:
            vol = vol/self.stdev       
    
        thresh = threshold_otsu(montage(vol,grid_shape=(1,vol.shape[0])))
        mask = (vol>thresh/4).astype(float) 
        #Divide by 4 is necessary to avoid brain regions in background. But, this may be hacky. Fix it?

        vol = torch.unsqueeze(torch.from_numpy(vol),dim=0)
        mask = torch.unsqueeze(torch.from_numpy(mask),dim=0)
        return vol.float(),mask.float(),filename

