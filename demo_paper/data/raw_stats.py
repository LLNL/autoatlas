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

data_files = ['/p/lustre1/hcpdata/processed/T1_decimate/2mm/{}_T1w_brain_2.nii.gz'.format(smpl) for smpl in samples]
#data_files = ['/p/lustre1/mohan3/Data/TBI/HCP/2mm/train_nm/{}/T1.nii.gz'.format(smpl) for smpl in samples]

mean = 0.0
stdev = 0.0
minm = 0.0
maxm = 0.0
for dfilen in data_files:
    data = nib.load(dfilen).get_fdata() 

    thresh = threshold_otsu(montage(data,grid_shape=(1,data.shape[0])))
    mask = (data <= thresh/1.5)
    labs = cc3d.connected_components(mask)
    uniqs = np.unique(labs)
    max_counts,max_label = 0,0
    for u in uniqs:
        cnt = np.sum(np.bitwise_and(labs==u,mask==True))
        if cnt > max_counts:
            max_counts = cnt
            max_label = u
    mask = np.bitwise_not(labs==max_label)
    
    #import matplotlib.pyplot as plt
    #plt.imshow(mask[mask.shape[0]//2])
    #plt.show()   
    #plt.imshow(mask[:,mask.shape[1]//2])
    #plt.show()   
    #plt.imshow(mask[:,:,mask.shape[2]//2])
    #plt.show()   
 
    mdata = data[mask]
    smpl_mean,smpl_std = mdata.mean(),mdata.std()
    mean += smpl_mean
    stdev += smpl_std
    smpl_min,smpl_max = mdata.min(),mdata.max()
    minm += smpl_min
    maxm += smpl_max

#    minm = smpl_min if smpl_min<minm else minm
#    maxm = smpl_max if smpl_max>maxm else maxm
    
    print(smpl_mean,smpl_std,smpl_min,smpl_max)

mean = mean/len(data_files)
stdev = stdev/len(data_files)
minm = minm/len(data_files)
maxm = maxm/len(data_files)
print('Number of training files is {}'.format(len(data_files)))

print("mean: " + str(mean))
print("stdev: " + str(stdev))
print("max: " + str(maxm))
print("min: " + str(minm))
