import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

achans = [4,16]
test_start = 0
num_test = 6

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05'
num_labels = 16
smpl_label = None

for ac in achans:
    out_folder = os.path.join(log_dir.format(ac),'paper','labrec')
    test_folder = os.path.join(log_dir.format(ac),'test_aa')
    test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-3:]=='.h5']
    test_files.sort()

    if not os.path.exists(out_folder):
        os.makedirs(out_folder) 

    def save_fig(arr,filen,cmap=None):
        plt.imshow(arr,cmap=cmap)
        plt.axis('off')
        plt.colorbar()
        plt.savefig(os.path.join(out_folder,filen),bbox_inches='tight')
        plt.close()

    for use_mask in [True,False]:
        for i in range(test_start,test_start+num_test):
            with h5py.File(test_files[i],'r') as f:
                seg = np.array(f['segmentation'])
                rec = np.array(f['reconstruction'])
                mk = np.array(f['mask'])
            if use_mask:
                rec = rec*seg*mk[np.newaxis]
            #else:    
            #    rec = rec*seg
            sh = rec.shape
            if smpl_label is None:
                for nlab in range(rec.shape[0]):
                    save_fig(rec[nlab,:,:,sh[2]//2],'rec{}_lab{}_mk{}_x.png'.format(i,nlab,int(use_mask)),cmap='gray')
            else:
                save_fig(rec[nlab,:,:,sh[2]//2],'rec{}_lab{}_mk{}_x.png'.format(i,smpl_label,int(use_mask)),cmap='gray')
                
