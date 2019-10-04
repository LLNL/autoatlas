#import matplotlib.pyplot as plt
from segmenter import AutoSegmenter
from data import HCPDataset
#from plot import stack_plot
import os

#Regularization parameters
entr_reg = 0.0
smooth_reg = 0.1
devr_reg = 1.0
min_freqs = 0.05 

#Parameters
checkpoint_dir='./checkpoints_smooth0.1/'
#learning_rate = 1e-4
learning_rate = 1e-5 #after 38 iterations
num_epochs = 100
load_epoch = 52
num_labels = 16
dims = [80,96,80]
#dims = [176,208,176]
mean = 135.4005
stdev = 286.3180
#minm,maxm = 0,2323.2244
#[160,160,160] is probably the maximum
#data_folder = '/usr/workspace/wsb/tbidata/workspace_aditya/Data/ss_nii/'
train_folder = '/p/lustre3/kaplan7/T1_decimate/2mm/train'
test_folder = '/p/lustre3/kaplan7/T1_decimate/2mm/test'

#Datasets
train_files = [os.path.join(train_folder,f) for f in os.listdir(train_folder) if f[-7:]=='.nii.gz']
test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-7:]=='.nii.gz']

train_data = HCPDataset(train_files,dims,mean,stdev)
valid_data = HCPDataset(test_files,dims,mean,stdev)

#NN Model
#autoseg = AutoSegmenter(num_labels,smooth_reg=1000.0,unif_reg=100.0,entr_reg=100.0,batch=2,eps=1e-15,lr=1e-4,device='cuda')
if load_epoch >= 0:
    autoseg = AutoSegmenter(num_labels,smooth_reg=smooth_reg,devr_reg=devr_reg,entr_reg=entr_reg,min_freqs=min_freqs,batch=2,lr=learning_rate,device='cuda',checkpoint_dir=checkpoint_dir,load_checkpoint_epoch=load_epoch)
elif load_epoch == -1:
    autoseg = AutoSegmenter(num_labels,smooth_reg=smooth_reg,devr_reg=devr_reg,entr_reg=entr_reg,min_freqs=min_freqs,batch=2,lr=learning_rate,device='cuda',checkpoint_dir=checkpoint_dir)
else:
    raise ValueError('load_epoch must be greater than or equal to -1')

#Training
for epoch in range(load_epoch+1,num_epochs):
    print("Epoch {}".format(epoch))
    autoseg.train(train_data)
    autoseg.test(valid_data)
    autoseg.checkpoint(epoch)
