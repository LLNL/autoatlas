import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt

tag_list = ['mxvol_labs16_relr0_0_smooth0_005_devrr0_1_devrm0_9','mxvol_labs16_smooth0_0_devrr0_1_devrm0_9','mxvol_labs16_smooth0_005_devrr0_0_devrm0_9','mxvol_labs16_smooth0_005_devrr0_1_devrm0_9']
save_dirs = ['no_rel','no_smooth','no_devr','all_loss']
subjs = ['131924']
num_labels = 16

rgb_list = np.array([[128,0,0],[0,0,128],[128,128,0],[0,128,128],[128,128,128],[255,255,255],[170,110,40],[230,25,75],[245,130,48],[255,255,25],[210,245,60],[60,180,75],[70,240,240],[0,130,200],[145,30,180],[240,50,230]]).astype(np.uint8)[:num_labels]

def save_fig(arr,filen,cmap=None,out_dir='./'):
    plt.imshow(arr,cmap=cmap)
    plt.axis('off')
    #if len(arr.shape)<3:
    #    plt.colorbar()
    plt.savefig(os.path.join(out_dir,filen),bbox_inches='tight')
    plt.close()

for tidx,tag in enumerate(tag_list):
    for sub in subjs:
        out_folder = os.path.join(os.path.join('figs',save_dirs[tidx]),sub)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder) 

        log_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/{}/test_mx/{}/'.format(tag,sub)
        seg = nib.load(os.path.join(log_dir,'seg_vol.nii.gz')).get_fdata()
        rec = nib.load(os.path.join(log_dir,'rec_vol.nii.gz')).get_fdata()
        prob = nib.load(os.path.join(log_dir,'prob_vol.nii.gz')).get_fdata()
        log_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/{}/test_mx/{}/'.format('',sub)
        T1 = nib.load(os.path.join(log_dir,'T1.nii.gz')).get_fdata()
        mask = nib.load(os.path.join(log_dir,'mask.nii.gz')).get_fdata()

        segrgb = np.zeros((seg.shape[0],seg.shape[1],seg.shape[2],3),dtype=np.uint8)
        for i in range(num_labels):
            segrgb[seg==i] = rgb_list[i]
        segrgb[mask==0.0] = np.array([0,0,0],dtype=np.uint8)
        
        segrgb = np.transpose(segrgb,axes=(2,0,1,3))[::-1]
        sh = seg.shape
        save_fig(segrgb[sh[0]//2],'seg_z.png',out_dir=out_folder)
        save_fig(segrgb[:,sh[1]//2],'seg_y.png',out_dir=out_folder)
        save_fig(segrgb[:,:,sh[2]//2],'seg_x.png',out_dir=out_folder)
        
        assert prob.shape == rec.shape and prob.ndim == 4
        rec = np.sum(rec*prob,axis=-1)
        rec[mask==0.0] = 0.0 
        rec = np.transpose(rec,axes=(2,0,1))[::-1]
        sh = rec.shape
        save_fig(rec[sh[0]//2],'rec_z.png',out_dir=out_folder)
        save_fig(rec[:,sh[1]//2],'rec_y.png',out_dir=out_folder)
        save_fig(rec[:,:,sh[2]//2],'rec_x.png',out_dir=out_folder)
