import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from skimage.feature import canny

achans = [4,16]
test_start = 4
num_test = 1
alpha = 0.5

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05'
out_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05'
num_labels = 16
smpl_label = None

for ac in achans:
    out_folder = os.path.join(out_dir,'supp','recs')
    test_folder = os.path.join(log_dir.format(ac),'test_aa')
    test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-3:]=='.h5']
    test_files.sort()

    if not os.path.exists(out_folder):
        os.makedirs(out_folder) 

    def save_fig(arr,filen,cmap=None,vmin=None,vmax=None):
        plt.imshow(arr,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.axis('off')
        plt.savefig(os.path.join(out_folder,filen),bbox_inches='tight')
        plt.close()

    for i in range(test_start,test_start+num_test):
        with h5py.File(test_files[i],'r') as f:
            gt = np.array(f['ground_truth'])
            seg = np.array(f['segmentation'])
            rec = np.array(f['reconstruction'])
            mk = np.array(f['mask'])

        maxval = np.max(gt)
        sh = rec.shape
        foreerrs,backerrs = [],[]
        for nlab in range(sh[0]):
            vol = rec[nlab]
            foreg_mask = seg[nlab]*mk 
            backg_mask = (1-seg[nlab])*mk
            
            foreerrs.append(np.sum((vol-gt)*foreg_mask*(vol-gt))/np.sum(foreg_mask))            
            backerrs.append(np.sum((vol-gt)*backg_mask*(vol-gt))/np.sum(backg_mask)) 
 
            img = vol[:,:,sh[3]//2]/maxval
            color_img = np.zeros((img.shape[0],img.shape[1],3),dtype=float)
            color_img[:,:,:] = img[:,:,np.newaxis]
            color_img[canny(mk[:,:,sh[3]//2])] = np.array([1,0,0])   
            color_img[canny(seg[nlab,:,:,sh[3]//2])] = np.array([0,0,1])   
            save_fig((color_img*255).astype(np.uint8),'rec{}_ac{}_lab{}_x.png'.format(i,ac,nlab),cmap='gray')
        
        print('C_a = {}'.format(ac))
        strg = ''
        for nlab in range(sh[0]):
            strg += '{:.3f}/{:.3f} &'.format(foreerrs[nlab],backerrs[nlab]) 
        print(strg)   
 
        gt = gt[:,:,sh[3]//2]
        rec = rec[:,:,:,sh[3]//2]
        seg = seg[:,:,:,sh[3]//2]
        mk = mk[:,:,sh[3]//2]
        save_fig(gt,'gt{}_ac{}_x.png'.format(i,ac),cmap='gray',vmin=0,vmax=maxval)
        rec = np.sum(rec*seg*mk[np.newaxis],axis=0)
        save_fig(rec,'rec{}_ac{}_x.png'.format(i,ac),cmap='gray',vmin=0,vmax=maxval)
            
