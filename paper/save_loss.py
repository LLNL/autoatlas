import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('../')
from utils import get_config
from segmenter import AutoSegmenter
from data import HCPDataset
import argparse

#log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05'
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir',type=str,default='./checkpoints/',help='Directory for storing run time data')
ARGS = parser.parse_args()
log_dir = ARGS.log_dir

num_labels,data_chan,space_dim,unet_chan,unet_blocks,aenc_chan,aenc_depth,num_epochs,batch,re_pow,lr,smooth_reg,devr_reg,min_freqs,train_folder,test_folder,stdev,size_dim = get_config(log_dir+'/args.cfg')

train_files = [os.path.join(train_folder,f) for f in os.listdir(train_folder) if f[-7:]=='.nii.gz']
dims = [size_dim,size_dim,size_dim]
train_data = HCPDataset(train_files,dims,None,stdev)

test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-7:]=='.nii.gz']
dims = [size_dim,size_dim,size_dim]
test_data = HCPDataset(test_files,dims,None,stdev)

train_tot,train_mse,train_smooth,train_devr = [],[],[],[]
test_tot,test_mse,test_smooth,test_devr = [],[],[],[]
for load_epoch in range(num_epochs):
    if not os.path.exists(log_dir+'/model_epoch_{}.pth'.format(load_epoch)):
        print('Checkpoint at epoch {} is not found. Stopped logging'.format(load_epoch))
        break
    autoseg = AutoSegmenter(num_labels,sizes=dims,data_chan=data_chan,smooth_reg=smooth_reg,devr_reg=devr_reg,entr_reg=0.0,min_freqs=min_freqs,batch=batch,lr=lr,unet_chan=unet_chan,unet_blocks=unet_blocks,aenc_chan=aenc_chan,aenc_depth=aenc_depth,re_pow=re_pow,device='cuda',checkpoint_dir=log_dir,load_checkpoint_epoch=load_epoch)
   
#    tot,mse,smooth,devr = autoseg.test(train_data)
#    train_tot.append(tot)
#    train_mse.append(mse)
#    train_smooth.append(smooth)
#    train_devr.append(devr) 
    
    tot,mse,smooth,devr = autoseg.test(test_data)
    test_tot.append(tot)
    test_mse.append(mse)
    test_smooth.append(smooth)
    test_devr.append(devr) 

    train_tot.append(autoseg.train_tot_loss)
    train_mse.append(autoseg.train_mse_loss)
    train_smooth.append(autoseg.train_smooth_loss)
    train_devr.append(autoseg.train_devr_loss)
    #test_tot.append(autoseg.test_tot_loss)
    #test_mse.append(autoseg.test_mse_loss)
    #test_smooth.append(autoseg.test_smooth_loss)
    #test_devr.append(autoseg.test_devr_loss)

if not os.path.exists(os.path.join(log_dir,'paper')):
    os.makedirs(os.path.join(log_dir,'paper'))
 
np.save(os.path.join(log_dir,'paper','train_tot_loss.npy'),np.array(train_tot)) 
np.save(os.path.join(log_dir,'paper','train_mse_loss.npy'),np.array(train_mse)) 
np.save(os.path.join(log_dir,'paper','train_smooth_loss.npy'),np.array(train_smooth)) 
np.save(os.path.join(log_dir,'paper','train_devr_loss.npy'),np.array(train_devr)) 

np.save(os.path.join(log_dir,'paper','test_tot_loss.npy'),np.array(test_tot)) 
np.save(os.path.join(log_dir,'paper','test_mse_loss.npy'),np.array(test_mse)) 
np.save(os.path.join(log_dir,'paper','test_smooth_loss.npy'),np.array(test_smooth)) 
np.save(os.path.join(log_dir,'paper','test_devr_loss.npy'),np.array(test_devr)) 
