import numpy as np
from data import HCPDataset
import os
import torch

train_folder = '/p/lustre3/kaplan7/T1_decimate/1mm/train'
train_files = [os.path.join(train_folder,f) for f in os.listdir(train_folder) if f[-7:]=='.nii.gz']
train_data = HCPDataset(train_files)

loader = torch.utils.data.DataLoader(train_data,batch_size=1,shuffle=False)

mean = 0.0
for data in loader:
    mean += data.mean()
mean = mean/len(train_data)

stdev = 0.0
for data in loader:
    stdev += ((data-mean)*(data-mean)).mean()
stdev = np.sqrt(stdev/len(train_data))

print("mean: " + str(mean))
print("stdev: " + str(stdev))

