#from scipy import ndimage
import nibabel as nib
import h5py
import numpy as np
from skimage.util import montage
from skimage.filters import threshold_otsu
from PIL import Image
import os
#import matplotlib.pyplot as plt
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
        mask = (vol>thresh/4).astype(float) #Divide by 4 is necessary to avoid brain regions in background. But, this may be hacky. Fix it?

        vol = torch.unsqueeze(torch.from_numpy(vol),dim=0)
        mask = torch.unsqueeze(torch.from_numpy(mask),dim=0)
        return vol.float(),mask.float()

class CelebDataset(Dataset):
    def __init__(self,folder,num=None,dims=None,mean=None,stdev=None,cmap='gray'):
        self.folder = folder
        self.dims = dims
        self.mean = mean
        self.stdev = stdev
        self.cmap = cmap
        image_folder = os.path.join(folder,'images')
        mask_folder = os.path.join(folder,'masks')
        self.image_files = [os.path.join(image_folder,f) for f in os.listdir(image_folder) if f[-4:]=='.jpg']
        if num is not None:
            self.image_files = self.image_files[:num]
        self.mask_files = [os.path.join(mask_folder,'mask_{}.png'.format(f[-9:-4])) for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self,idx):
        #assert self.image_files[idx][-9:-4]==self.mask_files[idx][-9:-4]
        #print(self.image_files[idx],self.mask_files[idx]) 

        image = Image.open(self.image_files[idx]) 
        mask = Image.open(self.mask_files[idx])
       
        if self.dims is not None:
            image = image.resize(self.dims,Image.BILINEAR)     
            mask = mask.resize(self.dims,Image.BILINEAR)

        image = np.array(image,dtype=np.float32)
        #Input mask type is background, but we need foreground mask
        mask = np.array(mask,dtype=np.float32)
        mask[mask<np.max(mask)] = 0

        mask_fore = np.zeros(mask.shape,dtype=np.float32) 
        mask_fore[mask==0] = 1 
 
        if self.mean is not None:
            image = image-self.mean
        if self.stdev is not None:
            image = image/self.stdev       
        else:
            image = image/255.0

        if self.cmap=='gray':
            image = torch.from_numpy(np.mean(image,axis=2))
            image = torch.unsqueeze(image,dim=0)
        else: 
            image = torch.from_numpy(image).permute(2,0,1)

#        plt.imshow(image[0])
#        plt.show()        

        mask_fore = torch.unsqueeze(torch.from_numpy(mask_fore),dim=0) 
        return image.float(),mask_fore.float()
         

