#import matplotlib.pyplot as plt
from segmenter import AutoSegmenter
from data import HCPDataset
#from plot import stack_plot
import os

#Parameters
num_epochs = 20
load_epoch = 9
num_labels = 16
dims = [80,96,80]
#dims = [176,208,176]
mean = 135.4005
stdev = 286.3180
#[160,160,160] is probably the maximum
#data_folder = '/usr/workspace/wsb/tbidata/workspace_aditya/Data/ss_nii/'
train_folder = '/p/lustre3/kaplan7/T1_decimate/2mm/train'
test_folder = '/p/lustre3/kaplan7/T1_decimate/2mm/test'

#Datasets
train_files = [os.path.join(train_folder,f) for f in os.listdir(train_folder) if f[-7:]=='.nii.gz']
test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-7:]=='.nii.gz']

train_data = HCPDataset(train_files,dims,mean,stdev)
valid_data = HCPDataset(test_files,dims,mean,stdev)

train_seg = HCPDataset(train_files[0],dims,mean)
valid_seg = HCPDataset(test_files[0],dims,stdev)

#NN Model
#autoseg = AutoSegmenter(num_labels,smooth_reg=1000.0,unif_reg=100.0,entr_reg=100.0,batch=2,eps=1e-15,lr=1e-4,device='cuda')
if load_epoch >= 0:
    autoseg = AutoSegmenter(num_labels,smooth_reg=0.0,unif_reg=0.0,entr_reg=0.0,batch=2,lr=1e-4,device='cuda',load_checkpoint_epoch=load_epoch)
elif load_epoch == -1:
    autoseg = AutoSegmenter(num_labels,smooth_reg=0.0,unif_reg=0.0,entr_reg=0.0,batch=2,lr=1e-4,device='cuda')
else:
    raise ValueError('load_epoch must be greater than or equal to -1')

#Training
for epoch in range(load_epoch+1,num_epochs):
    print("Epoch {}".format(epoch))
    autoseg.train(train_data)
    autoseg.test(valid_data)
    autoseg.checkpoint(epoch)
