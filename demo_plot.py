import matplotlib.pyplot as plt
from segmenter import AutoSegmenter
from data import SSDataset
from plot import stack_plot
import os

#Parameters
load_epoch = 26
num_labels = 8
dims = [128,128,128]
#[160,160,160] is probably the maximum
data_folder = '/usr/workspace/wsb/tbidata/workspace_aditya/Data/ss_nii/'

#Datasets
data_files = [os.path.join(data_folder,f) for f in os.listdir(data_folder) if f[-4:]=='.nii']

train_seg = SSDataset(data_files[-1],dims)
valid_seg = SSDataset(data_files[0],dims)

#NN Model
autoseg = AutoSegmenter(num_labels,smooth_reg=1000.0,unif_reg=100.0,batch=2,eps=1e-15,lr=1e-4,device='cuda',load_checkpoint_epoch=load_epoch)

#Training
tseg,tvol = autoseg.segment(train_seg)
vseg,vvol = autoseg.segment(valid_seg)
for i in range(num_labels):
    stack_plot([tvol[0,0],vvol[0,0]],[tseg[0,i],vseg[0,i]],'epoch_{}_label_{}_sample_1.png'.format(load_epoch,i))