import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image

tags = ['aatest_var']
save_dirs = ['aatest_var']
test_dirs = ['test_zn']
subjs = ['979984','305830']
num_labels = 16

rgb_list = np.array([[128,128,128],[70,240,240],[255,255,255],[230,25,75],[0,0,128],[128,128,0],[0,128,128],[170,110,40],[245,130,48],[255,255,25],[128,0,0],[210,245,60],[60,180,75],[0,130,200],[145,30,180],[240,50,230]]).astype(np.uint8)[:num_labels]

def save_fig(arr,filen,cmap=None,out_dir='./'):
    plt.imshow(arr,cmap=cmap)
    plt.axis('off')
    #if len(arr.shape)<3:
    #    plt.colorbar()
    plt.savefig(os.path.join(out_dir,filen),bbox_inches='tight')
    plt.close()

for tidx,tag in enumerate(tags):
    for sub in subjs:
        out_folder = os.path.join(os.path.join('figs',save_dirs[tidx]),sub)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder) 

        log_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/fMRI_var_flat_images/{}/{}/'.format(tag,test_dirs[tidx])
        seg = np.load(os.path.join(log_dir,'{}_seg_vol.npy'.format(sub)))
        rec = np.load(os.path.join(log_dir,'{}_rec_vol.npy'.format(sub)))
        prob = np.load(os.path.join(log_dir,'{}_prob_vol.npy'.format(sub)))
        log_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/fMRI_var_flat_images/proc_768x384/{}/'.format(test_dirs[tidx])
        fMRI = np.asarray(Image.open(os.path.join(log_dir,'{}_rfMRI_REST1_LR_var.tif'.format(sub))))
        mask = np.asarray(Image.open(os.path.join(log_dir,'{}_rfMRI_REST1_LR_var_mask.png'.format(sub))))

        segrgb = np.zeros((seg.shape[0],seg.shape[1],3),dtype=np.uint8)
        for i in range(num_labels):
            segrgb[seg==i] = rgb_list[i]
        segrgb[mask==0.0] = np.array([0,0,0],dtype=np.uint8)
        
        save_fig(segrgb,'seg.png',out_dir=out_folder)
        assert prob.shape == rec.shape and prob.ndim == 3
        rec = np.sum(rec*prob,axis=-1) 
        rec[mask==0.0] = 0
        save_fig(rec,'rec.png',out_dir=out_folder,cmap='gray')
        save_fig(fMRI,'gt.png',out_dir=out_folder,cmap='gray')
        save_fig(mask,'mask.png',out_dir=out_folder,cmap='gray')
