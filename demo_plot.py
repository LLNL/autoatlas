import matplotlib.pyplot as plt
from segmenter import AutoSegmenter
from data import HCPDataset
from plot import stack_plot,write_nifti
import os
import numpy as np

#Parameters
checkpoint_dir='./checkpoints_smooth0.1/'
num_epochs = 55
num_labels = 8
dims = [80,96,80]
mean = 135.4005
stdev = 286.3180

#train_folder = '/p/lustre3/kaplan7/T1_decimate/2mm/train'
#test_folder = '/p/lustre3/kaplan7/T1_decimate/2mm/test'
train_folder = '/p/gscratchr/mohan3/Data/T1_decimate/2mm/train'
test_folder = '/p/gscratchr/mohan3/Data/T1_decimate/2mm/test'

#Datasets
train_files = [os.path.join(train_folder,f) for f in os.listdir(train_folder) if f[-7:]=='.nii.gz']
test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-7:]=='.nii.gz']

train_tot,test_tot,train_mse,test_mse,train_smooth,test_smooth,train_entr,test_entr,train_devr,test_devr = [],[],[],[],[],[],[],[],[],[]
for load_epoch in range(num_epochs):
    autoseg = AutoSegmenter(num_labels,smooth_reg=0.0,devr_reg=0.0,entr_reg=0.0,batch=2,lr=1e-4,device='cuda',load_checkpoint_epoch=load_epoch,checkpoint_dir=checkpoint_dir)
    train_tot.append(autoseg.train_tot_loss)
    test_tot.append(autoseg.test_tot_loss)
    train_mse.append(autoseg.train_mse_loss)
    test_mse.append(autoseg.test_mse_loss)
    train_smooth.append(autoseg.train_smooth_loss)
    test_smooth.append(autoseg.test_smooth_loss)
    train_entr.append(autoseg.train_entr_loss)
    test_entr.append(autoseg.test_entr_loss)
    train_devr.append(autoseg.train_devr_loss)
    test_devr.append(autoseg.test_devr_loss)

plt.plot(range(num_epochs),train_tot)
plt.plot(range(num_epochs),test_tot)
plt.yscale('log'); 
plt.xlabel('epochs'); plt.ylabel('tot loss')
plt.legend(['train','test'])
plt.savefig(checkpoint_dir+'/tot_loss.png')
plt.close()

plt.plot(range(num_epochs),train_mse)
plt.plot(range(num_epochs),test_mse)
plt.yscale('log'); 
plt.xlabel('epochs'); plt.ylabel('mse loss')
plt.legend(['train','test'])
plt.savefig(checkpoint_dir+'/mse_loss.png')
plt.close()

plt.plot(range(num_epochs),train_smooth)
plt.plot(range(num_epochs),test_smooth)
plt.yscale('log'); 
plt.xlabel('epochs'); plt.ylabel('smooth loss')
plt.legend(['train','test'])
plt.savefig(checkpoint_dir+'/smooth_loss.png')
plt.close()

plt.plot(range(num_epochs),train_entr)
plt.plot(range(num_epochs),test_entr)
plt.yscale('log');
plt.xlabel('epochs'); plt.ylabel('entr loss')
plt.legend(['train','test'])
plt.savefig(checkpoint_dir+'/entr_loss.png')
plt.close()

plt.plot(range(num_epochs),train_devr)
plt.plot(range(num_epochs),test_devr)
plt.yscale('log');
plt.xlabel('epochs'); plt.ylabel('devr loss')
plt.legend(['train','test'])
plt.savefig(checkpoint_dir+'/devr_loss.png')
plt.close()

train_data = HCPDataset(train_files[:10],dims,mean,stdev)
test_data = HCPDataset(test_files[:10],dims,mean,stdev)

test_seg,test_rec,test_vol = autoseg.segment(test_data)
test_auto,_ = autoseg.classrec(test_data)

for i in range(10): 
    stack_plot(np.stack([test_vol[i,0],test_rec[i,0]],axis=0),checkpoint_dir+'/gtvsrec_z_{}.png'.format(i),sldim='z',nrows=1)
    stack_plot(np.stack([test_vol[i,0],test_rec[i,0]],axis=0),checkpoint_dir+'/gtvsrec_y_{}.png'.format(i),sldim='y',nrows=1)
    stack_plot(np.stack([test_vol[i,0],test_rec[i,0]],axis=0),checkpoint_dir+'/gtvsrec_x_{}.png'.format(i),sldim='x',nrows=1)
    stack_plot(test_seg[i],checkpoint_dir+'/seg_z_{}.png'.format(i),sldim='z',nrows=2)
    stack_plot(test_seg[i],checkpoint_dir+'/seg_y_{}.png'.format(i),sldim='y',nrows=2)
    stack_plot(test_seg[i],checkpoint_dir+'/seg_x_{}.png'.format(i),sldim='x',nrows=2)
    #write_nifti(test_seg[i],checkpoint_dir+'/seg_{}.nii.gz'.format(i))
    stack_plot(test_auto[i],checkpoint_dir+'/auto_z_{}.png'.format(i),sldim='z',nrows=2)
    stack_plot(test_auto[i],checkpoint_dir+'/auto_y_{}.png'.format(i),sldim='y',nrows=2)
    stack_plot(test_auto[i],checkpoint_dir+'/auto_x_{}.png'.format(i),sldim='x',nrows=2)

test_seg,test_rec,test_vol = autoseg.segment(test_data,masked=True)
test_auto,_ = autoseg.classrec(test_data,masked=True)
for i in range(10):
    stack_plot(np.stack([test_vol[i,0],test_rec[i,0]],axis=0),checkpoint_dir+'/mk_gtvsrec_z_{}.png'.format(i),sldim='z',nrows=1)
    stack_plot(np.stack([test_vol[i,0],test_rec[i,0]],axis=0),checkpoint_dir+'/mk_gtvsrec_y_{}.png'.format(i),sldim='y',nrows=1)
    stack_plot(np.stack([test_vol[i,0],test_rec[i,0]],axis=0),checkpoint_dir+'/mk_gtvsrec_x_{}.png'.format(i),sldim='x',nrows=1)
    stack_plot(test_seg[i],checkpoint_dir+'/mk_seg_z_{}.png'.format(i),sldim='z',nrows=2)
    stack_plot(test_seg[i],checkpoint_dir+'/mk_seg_y_{}.png'.format(i),sldim='y',nrows=2)
    stack_plot(test_seg[i],checkpoint_dir+'/mk_seg_x_{}.png'.format(i),sldim='x',nrows=2)
    #write_nifti(test_seg[i],checkpoint_dir+'/seg_{}.nii.gz'.format(i))
    stack_plot(test_auto[i],checkpoint_dir+'/mk_auto_z_{}.png'.format(i),sldim='z',nrows=2)
    stack_plot(test_auto[i],checkpoint_dir+'/mk_auto_y_{}.png'.format(i),sldim='y',nrows=2)
    stack_plot(test_auto[i],checkpoint_dir+'/mk_auto_x_{}.png'.format(i),sldim='x',nrows=2)
 
