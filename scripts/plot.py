import matplotlib.pyplot as plt
from segmenter import AutoSegmenter
from data import HCPDataset,CelebDataset
from plot_fun import stack_plot
import os
import numpy as np
import argparse
import os
from utils import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir',type=str,default='./checkpoints/',help='Directory for storing run time data')
parser.add_argument('--num_test',type=int,default=1,help='Number of samples to record')
ARGS = parser.parse_args()
    
num_labels,data_chan,space_dim,unet_chan,unet_blocks,aenc_chan,aenc_depth,num_epochs,batch,re_pow,lr,smooth_reg,devr_reg,min_freqs,train_folder,test_folder,stdev,size_dim = get_config(ARGS.log_dir+'/args.cfg')

#Datasets
if space_dim==3:
    test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-7:]=='.nii.gz'][:ARGS.num_test]
    dims = [size_dim,size_dim,size_dim]
    test_data = HCPDataset(test_files,dims,None,stdev)
elif space_dim==2:
    dims = [size_dim,size_dim]
    cmap = 'gray' if data_chan==1 else 'rgb'
    test_data = CelebDataset(test_folder,num=ARGS.num_test,dims=dims,mean=None,stdev=stdev,cmap=cmap) 
else:
    raise ValueError('Argument space_dim must be either 2 or 3')

train_tot,test_tot,train_mse,test_mse,train_smooth,test_smooth,train_devr,test_devr = [],[],[],[],[],[],[],[]
for load_epoch in range(num_epochs):
    if not os.path.exists(ARGS.log_dir+'/model_epoch_{}.pth'.format(load_epoch)):
        print('Checkpoint at epoch {} is not found. Stopped logging'.format(load_epoch))
        break  
    autoseg = AutoSegmenter(num_labels,sizes=dims,data_chan=data_chan,smooth_reg=smooth_reg,devr_reg=devr_reg,entr_reg=0.0,min_freqs=min_freqs,batch=batch,lr=lr,unet_chan=unet_chan,unet_blocks=unet_blocks,aenc_chan=aenc_chan,aenc_depth=aenc_depth,re_pow=re_pow,device='cuda',checkpoint_dir=ARGS.log_dir,load_checkpoint_epoch=load_epoch)
    train_tot.append(autoseg.train_tot_loss)
    test_tot.append(autoseg.test_tot_loss)
    train_mse.append(autoseg.train_mse_loss)
    test_mse.append(autoseg.test_mse_loss)
    train_smooth.append(autoseg.train_smooth_loss)
    test_smooth.append(autoseg.test_smooth_loss)
    train_devr.append(autoseg.train_devr_loss)
    test_devr.append(autoseg.test_devr_loss)

plt.plot(train_tot)
plt.plot(test_tot)
plt.yscale('log'); 
plt.xlabel('epochs'); plt.ylabel('tot loss')
plt.legend(['train','test'])
plt.savefig(ARGS.log_dir+'/tot_loss.png')
plt.close()

plt.plot(train_mse)
plt.plot(test_mse)
plt.yscale('log'); 
plt.xlabel('epochs'); plt.ylabel('mse loss')
plt.legend(['train','test'])
plt.savefig(ARGS.log_dir+'/mse_loss.png')
plt.close()

plt.plot(train_smooth)
plt.plot(test_smooth)
plt.yscale('log'); 
plt.xlabel('epochs'); plt.ylabel('smooth loss')
plt.legend(['train','test'])
plt.savefig(ARGS.log_dir+'/smooth_loss.png')
plt.close()

plt.plot(train_devr)
plt.plot(test_devr)
plt.yscale('log');
plt.xlabel('epochs'); plt.ylabel('devr loss')
plt.legend(['train','test'])
plt.savefig(ARGS.log_dir+'/devr_loss.png')
plt.close()

#rgb_base = np.array([np.random.rand()*255,np.random.rand()*255,np.random.rand()*255])
#rgb_list = rgb_base[np.newaxis]+(np.arange(0,num_labels)*(256/num_labels))[:,np.newaxis]
#rgb_list = np.mod(rgb_list,256).astype(np.uint8)
rgb_list = np.array([[128,0,0],[170,110,40],[128,128,0],[0,128,128],[0,0,128],[255,255,255],[230,25,75],[245,130,48],[255,255,25],[210,245,60],[60,180,75],[70,240,240],[0,130,200],[145,30,180],[240,50,230],[128,128,128]]).astype(np.uint8)[:num_labels]

test_seg,test_auto,test_vol = autoseg.segment(test_data)
test_auto = test_auto*test_seg
test_rec = np.sum(test_auto,axis=1)
if space_dim==3:
    test_seg = test_seg[:,:,0]
    test_auto = test_auto[:,:,0]
    test_vol = np.squeeze(test_vol)
    test_rec = np.squeeze(test_rec)
    for i in range(ARGS.num_test): 
        stack_plot(np.stack([test_vol[i],test_rec[i]],axis=0),ARGS.log_dir+'/gtvsrec_z_{}.png'.format(i),sldim='z',nrows=1)
        stack_plot(np.stack([test_vol[i],test_rec[i]],axis=0),ARGS.log_dir+'/gtvsrec_y_{}.png'.format(i),sldim='y',nrows=1)
        stack_plot(np.stack([test_vol[i],test_rec[i]],axis=0),ARGS.log_dir+'/gtvsrec_x_{}.png'.format(i),sldim='x',nrows=1)
        stack_plot(test_seg[i],ARGS.log_dir+'/seg_z_{}.png'.format(i),sldim='z',nrows=2)
        stack_plot(test_seg[i],ARGS.log_dir+'/seg_y_{}.png'.format(i),sldim='y',nrows=2)
        stack_plot(test_seg[i],ARGS.log_dir+'/seg_x_{}.png'.format(i),sldim='x',nrows=2)
        rgb_seg = np.sum(test_seg[i][:,:,:,:,np.newaxis]*rgb_list[:,np.newaxis,np.newaxis,np.newaxis],axis=0).astype(np.uint8)
        stack_plot([test_vol[i],rgb_seg],ARGS.log_dir+'/rgb_seg_z_{}.png'.format(i),sldim='z',nrows=1)
        stack_plot([test_vol[i],rgb_seg],ARGS.log_dir+'/rgb_seg_y_{}.png'.format(i),sldim='y',nrows=1)
        stack_plot([test_vol[i],rgb_seg],ARGS.log_dir+'/rgb_seg_x_{}.png'.format(i),sldim='x',nrows=1)
        #write_nifti(test_seg[i],ARGS.log_dir+'/seg_{}.nii.gz'.format(i))
        stack_plot(test_auto[i],ARGS.log_dir+'/auto_z_{}.png'.format(i),sldim='z',nrows=2)
        stack_plot(test_auto[i],ARGS.log_dir+'/auto_y_{}.png'.format(i),sldim='y',nrows=2)
        stack_plot(test_auto[i],ARGS.log_dir+'/auto_x_{}.png'.format(i),sldim='x',nrows=2)
else:
    test_vol = np.transpose(test_vol,(0,2,3,1))
    test_rec = np.transpose(test_rec,(0,2,3,1))
    test_seg = test_seg[:,:,0]
    test_auto = np.transpose(test_auto,(0,1,3,4,2))
    for i in range(ARGS.num_test): 
        stack_plot(np.stack([np.squeeze(test_vol[i]),np.squeeze(test_rec[i])],axis=0),ARGS.log_dir+'/gtvsrec_{}.png'.format(i),nrows=1)
        stack_plot(test_seg[i],ARGS.log_dir+'/seg_{}.png'.format(i),nrows=2)
        rgb_seg = np.sum(test_seg[i][:,:,:,np.newaxis]*rgb_list[:,np.newaxis,np.newaxis],axis=0).astype(np.uint8)
        stack_plot([np.squeeze(test_vol[i]),rgb_seg],ARGS.log_dir+'/rgb_seg_{}.png'.format(i),nrows=1)
        stack_plot(np.squeeze(test_auto[i]),ARGS.log_dir+'/auto_{}.png'.format(i),nrows=2)

test_seg,test_auto,test_vol = autoseg.segment(test_data,masked=True)
test_auto = test_auto*test_seg
test_rec = np.sum(test_auto,axis=1)
if space_dim==3:
    test_seg = test_seg[:,:,0]
    test_auto = test_auto[:,:,0] 
    test_vol = np.squeeze(test_vol)
    test_rec = np.squeeze(test_rec)
    for i in range(ARGS.num_test):
        stack_plot(np.stack([test_vol[i],test_rec[i]],axis=0),ARGS.log_dir+'/mk_gtvsrec_z_{}.png'.format(i),sldim='z',nrows=1)
        stack_plot(np.stack([test_vol[i],test_rec[i]],axis=0),ARGS.log_dir+'/mk_gtvsrec_y_{}.png'.format(i),sldim='y',nrows=1)
        stack_plot(np.stack([test_vol[i],test_rec[i]],axis=0),ARGS.log_dir+'/mk_gtvsrec_x_{}.png'.format(i),sldim='x',nrows=1)
        stack_plot(test_seg[i],ARGS.log_dir+'/mk_seg_z_{}.png'.format(i),sldim='z',nrows=2)
        stack_plot(test_seg[i],ARGS.log_dir+'/mk_seg_y_{}.png'.format(i),sldim='y',nrows=2)
        stack_plot(test_seg[i],ARGS.log_dir+'/mk_seg_x_{}.png'.format(i),sldim='x',nrows=2)
        #write_nifti(test_seg[i],ARGS.log_dir+'/seg_{}.nii.gz'.format(i))
        rgb_seg = np.sum(test_seg[i][:,:,:,:,np.newaxis]*rgb_list[:,np.newaxis,np.newaxis,np.newaxis],axis=0).astype(np.uint8)
        stack_plot([test_vol[i],rgb_seg],ARGS.log_dir+'/mk_rgb_seg_z_{}.png'.format(i),sldim='z',nrows=1)
        stack_plot([test_vol[i],rgb_seg],ARGS.log_dir+'/mk_rgb_seg_y_{}.png'.format(i),sldim='y',nrows=1)
        stack_plot([test_vol[i],rgb_seg],ARGS.log_dir+'/mk_rgb_seg_x_{}.png'.format(i),sldim='x',nrows=1)
        stack_plot(test_auto[i],ARGS.log_dir+'/mk_auto_z_{}.png'.format(i),sldim='z',nrows=2)
        stack_plot(test_auto[i],ARGS.log_dir+'/mk_auto_y_{}.png'.format(i),sldim='y',nrows=2)
        stack_plot(test_auto[i],ARGS.log_dir+'/mk_auto_x_{}.png'.format(i),sldim='x',nrows=2)
else:
    test_vol = np.transpose(test_vol,(0,2,3,1))
    test_rec = np.transpose(test_rec,(0,2,3,1))
    test_seg = test_seg[:,:,0]
    test_auto = np.transpose(test_auto,(0,1,3,4,2))
    for i in range(ARGS.num_test): 
        stack_plot(np.stack([np.squeeze(test_vol[i]),np.squeeze(test_rec[i])],axis=0),ARGS.log_dir+'/mk_gtvsrec_{}.png'.format(i),nrows=1)
        stack_plot(test_seg[i],ARGS.log_dir+'/mk_seg_{}.png'.format(i),nrows=2)
        rgb_seg = np.sum(test_seg[i][:,:,:,np.newaxis]*rgb_list[:,np.newaxis,np.newaxis],axis=0).astype(np.uint8)
        stack_plot([np.squeeze(test_vol[i]),rgb_seg],ARGS.log_dir+'/mk_rgb_seg_{}.png'.format(i),nrows=1)
        stack_plot(np.squeeze(test_auto[i]),ARGS.log_dir+'/mk_auto_{}.png'.format(i),nrows=2)
     