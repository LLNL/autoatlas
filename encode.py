import argparse
from segmenter import AutoSegmenter
from data import HCPDataset
from utils import get_config
import matplotlib.pyplot as plt
import os
import numpy as np
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir',type=str,default='./checkpoints/',help='Directory for storing run time data')
parser.add_argument('--load_epoch',type=int,required=True,help='Model epoch to load')
ARGS = parser.parse_args()

num_labels,data_chan,space_dim,unet_chan,unet_blocks,aenc_chan,aenc_depth,num_epochs,batch,re_pow,lr,smooth_reg,devr_reg,min_freqs,train_folder,test_folder,stdev,size_dim = get_config(ARGS.log_dir+'/args.cfg')

dims = [size_dim,size_dim,size_dim]

train_files = [os.path.join(train_folder,f) for f in os.listdir(train_folder) if f[-7:]=='.nii.gz']
train_data = HCPDataset(train_files,dims,None,stdev,ret_filen=True)

test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-7:]=='.nii.gz']
test_data = HCPDataset(test_files,dims,None,stdev,ret_filen=True)

autoseg = AutoSegmenter(num_labels,sizes=dims,data_chan=data_chan,smooth_reg=smooth_reg,devr_reg=devr_reg,entr_reg=0.0,min_freqs=min_freqs,batch=batch,lr=lr,unet_chan=unet_chan,unet_blocks=unet_blocks,aenc_chan=aenc_chan,aenc_depth=aenc_depth,re_pow=re_pow,device='cuda',checkpoint_dir=ARGS.log_dir,load_checkpoint_epoch=ARGS.load_epoch)

autoseg.eval(train_data,os.path.join(ARGS.log_dir,'train_aa'),ret_data=False)
autoseg.eval(test_data,os.path.join(ARGS.log_dir,'test_aa'),ret_data=False)

