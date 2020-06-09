import numpy as np
from autoatlas.data import NibData
import os
import torch
import csv
from skimage.filters import threshold_otsu
from skimage.util import montage
import cc3d
import nibabel as nib

samples = [] 
with open('/p/lustre1/mohan3/Data/TBI/HCP/2mm/train/subjects.txt','r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row)==1
        samples.append(row[0])

data_files = ['/p/lustre1/mohan3/Data/TBI/HCP/2mm/train_nm/{}/T1.nii.gz'.format(smpl) for smpl in samples]
mask_files = ['/p/lustre1/mohan3/Data/TBI/HCP/2mm/train_nm/{}/mask.nii.gz'.format(smpl) for smpl in samples]

mean = 0.0
stdev = 0.0
minm = np.inf
maxm = 0.0
for dfilen,mfilen in zip(data_files,mask_files):
    data = nib.load(dfilen).get_fdata() 
    mask = nib.load(mfilen).get_fdata() 

    mdata = data[mask==1.0]
    smpl_mean,smpl_std = mdata.mean(),mdata.std()
    mean += smpl_mean
    stdev += smpl_std
    smpl_min,smpl_max = mdata.min(),mdata.max()
    minm = smpl_min if smpl_min<minm else minm
    maxm = smpl_max if smpl_max>maxm else maxm
    
    print(smpl_mean,smpl_std,smpl_min,smpl_max)

mean = mean/len(data_files)
stdev = stdev/len(data_files)
print('Number of training files is {}'.format(len(data_files)))

print("mean: " + str(mean))
print("stdev: " + str(stdev))
print("max: " + str(maxm))
print("min: " + str(minm))
