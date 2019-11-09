import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05'
test_start = 1
num_labels = 16
num_test = 10
smpl_label = 5

out_folder = os.path.join(log_dir,'paper')
test_folder = os.path.join(log_dir,'test_aa')
test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-3:]=='.h5']

if not os.path.exists(out_folder):
    os.makedirs(out_folder) 

def save_fig(arr,filen,cmap=None):
    plt.imshow(arr,cmap=cmap)
    plt.axis('off')
    plt.savefig(os.path.join(out_folder,filen),bbox_inches='tight')
    plt.close()

for i in range(test_start,test_start+num_test):
    with h5py.File(test_files[i],'r') as f:
        seg = np.array(f['segmentation'])
        rec = np.array(f['reconstruction'])
        mk = np.array(f['mask'])
    rec = rec*seg*mk[np.newaxis]
    sh = rec.shape
    save_fig(rec[smpl_label,:,:,sh[2]//2],'rec{}_lab{}_x.png'.format(i,smpl_label),cmap='gray')
