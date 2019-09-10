import matplotlib.pyplot as plt
from segmenter import AutoSegmenter
from data import HCPDataset
from plot import stack_plot
import os
import numpy as np

#Parameters
num_epochs = 20
num_labels = 16
dims = [80,96,80]
mean = 135.4005
stdev = 286.3180

train_folder = '/p/lustre3/kaplan7/T1_decimate/2mm/train'
test_folder = '/p/lustre3/kaplan7/T1_decimate/2mm/test'

#Datasets
train_files = [os.path.join(train_folder,f) for f in os.listdir(train_folder) if f[-7:]=='.nii.gz']
test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-7:]=='.nii.gz']

train_data = HCPDataset(train_files[0],dims,mean,stdev)
test_data = HCPDataset(test_files[0],dims,mean,stdev)

train_loss,test_loss = [],[]
for load_epoch in range(num_epochs):
    autoseg = AutoSegmenter(num_labels,smooth_reg=0.0,unif_reg=0.0,entr_reg=0.0,batch=2,lr=1e-4,device='cuda',load_checkpoint_epoch=load_epoch)
    train_loss.append(autoseg.curr_train_loss)
    test_loss.append(autoseg.curr_test_loss)

plt.plot(range(num_epochs),train_loss)
plt.plot(range(num_epochs),test_loss)
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','test'])
plt.savefig('loss.png')

test_seg,test_rec,test_vol = autoseg.segment(test_data)
test_auto,_ = autoseg.classrec(test_data)

stack_plot(np.stack([test_vol[0,0],test_rec[0,0]],axis=0),'gtvsrec_z.png',sldim='z',nrows=1)
stack_plot(np.stack([test_vol[0,0],test_rec[0,0]],axis=0),'gtvsrec_y.png',sldim='y',nrows=1)
stack_plot(np.stack([test_vol[0,0],test_rec[0,0]],axis=0),'gtvsrec_x.png',sldim='x',nrows=1)
stack_plot(test_seg[0],'seg_z.png',sldim='z',nrows=2)
stack_plot(test_seg[0],'seg_y.png',sldim='y',nrows=2)
stack_plot(test_seg[0],'seg_x.png',sldim='x',nrows=2)
stack_plot(test_auto[0],'auto_z.png',sldim='z',nrows=2)
stack_plot(test_auto[0],'auto_y.png',sldim='y',nrows=2)
stack_plot(test_auto[0],'auto_x.png',sldim='x',nrows=2)

