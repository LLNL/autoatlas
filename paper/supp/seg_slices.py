import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

achan = 4
test_sample = 4

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05'
num_labels = 16

rgb_list = np.array([[128,0,0],[170,110,40],[128,128,0],[0,128,128],[0,0,128],[255,255,255],[230,25,75],[245,130,48],[255,255,25],[210,245,60],[60,180,75],[70,240,240],[0,130,200],[145,30,180],[240,50,230],[128,128,128]]).astype(np.uint8)[:num_labels]

def save_fig(arr,filen,cmap=None):
    plt.imshow(arr,cmap=cmap)
    plt.axis('off')
    #if len(arr.shape)<3:
    #    plt.colorbar()
    plt.savefig(os.path.join(out_folder,filen),bbox_inches='tight')
    plt.close()

out_folder = os.path.join(log_dir.format(achan),'supp','seg')
if not os.path.exists(out_folder):
    os.makedirs(out_folder) 

test_folder = os.path.join(log_dir.format(achan),'test_aa')
test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-3:]=='.h5']
test_files.sort()

i = test_sample
with h5py.File(test_files[i],'r') as f:
    gt = np.array(f['ground_truth'])
    seg = np.array(f['segmentation'])
    rec = np.array(f['reconstruction'])
    mk = np.array(f['mask'])
seg = seg*mk[np.newaxis]
rec = rec*mk[np.newaxis]
rec = np.sum(rec*seg,axis=0)
seg = np.sum(seg[:,:,:,:,np.newaxis]*rgb_list[:,np.newaxis,np.newaxis,np.newaxis],axis=0).astype(np.uint8)

sh = seg.shape
for j in range(0,sh[2],4):
    save_fig(gt[:,:,j],'gt{}_aenc{}_sl{}_x.png'.format(i,achan,j),cmap='gray')
    save_fig(seg[:,:,j],'seg{}_aenc{}_sl{}_x.png'.format(i,achan,j))


