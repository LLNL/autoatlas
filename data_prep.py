import numpy as np
from data import HCPDataset
import os
import torch

train_folder = '/p/gscratchr/mohan3/Data/T1_decimate/2mm/train'
train_files = [os.path.join(train_folder,f) for f in os.listdir(train_folder) if f[-7:]=='.nii.gz']
test_folder = '/p/gscratchr/mohan3/Data/T1_decimate/2mm/test'
test_files = [os.path.join(train_folder,f) for f in os.listdir(train_folder) if f[-7:]=='.nii.gz']

dataset = HCPDataset(train_files+test_files)
loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

def find_max(mask,max_ax,axis):
    ax_line = np.sum(mask,axis=axis)  
    mid = ax_line.size//2 
    ax_left = np.nonzero(ax_line[:mid]==0)[0]
    ax_right = np.nonzero(ax_line[mid:]==0)[0]
    ax_left = ax_left[-1] if len(ax_left)>0 else 0
    ax_right = ax_right[0] if len(ax_right)>0 else len(ax_line[mid:])
    len_ax = mid+ax_right-ax_left
    if len_ax > max_ax:
        max_ax = len_ax
    return max_ax

max_z,max_y,max_x = 0,0,0
for data,mask in loader:
    mask = np.squeeze(mask.cpu().numpy())
    max_z = find_max(mask,max_z,axis=(1,2))
    max_y = find_max(mask,max_z,axis=(0,2))
    max_x = find_max(mask,max_z,axis=(0,1))

print('max_z,max_y,max_x = {},{},{}'.format(max_z,max_y,max_x))
