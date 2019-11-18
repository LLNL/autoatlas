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

out_folder = os.path.join(log_dir.format(achan),'paper','seg')
if not os.path.exists(out_folder):
    os.makedirs(out_folder) 

test_folder = os.path.join(log_dir.format(achan),'test_aa')
test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-3:]=='.h5']
test_files.sort()

print(test_files[test_sample])
with h5py.File(test_files[test_sample],'r') as f:
    gt = np.array(f['ground_truth'])
    seg = np.array(f['segmentation'])
    rec = np.array(f['reconstruction'])
    mk = np.array(f['mask'])
segrgb = (seg[:,:,:,:,np.newaxis]*rgb_list[:,np.newaxis,np.newaxis,np.newaxis]).astype(np.uint8)

sh = rec.shape
save_fig(gt[sh[1]//2],'gt_mfig_z.png',cmap='gray')
save_fig(gt[:,sh[2]//2],'gt_mfig_y.png',cmap='gray')
save_fig(gt[:,:,sh[3]//2],'gt_mfig_x.png',cmap='gray')
for i in range(sh[0]):
    save_fig(rec[i,sh[1]//2],'rec{}_mfig_z.png'.format(i),cmap='gray')
    save_fig(rec[i,:,sh[2]//2],'rec{}_mfig_y.png'.format(i),cmap='gray')
    save_fig(rec[i,:,:,sh[3]//2],'rec{}_mfig_x.png'.format(i),cmap='gray')
    save_fig(segrgb[i,sh[1]//2],'seg{}_mfig_z.png'.format(i))
    save_fig(segrgb[i,:,sh[2]//2],'seg{}_mfig_y.png'.format(i))
    save_fig(segrgb[i,:,:,sh[3]//2],'seg{}_mfig_x.png'.format(i))

rec = rec*seg
for i in range(sh[0]):
    save_fig(rec[i,sh[1]//2],'recseg{}_mfig_z.png'.format(i),cmap='gray')
    save_fig(rec[i,:,sh[2]//2],'recseg{}_mfig_y.png'.format(i),cmap='gray')
    save_fig(rec[i,:,:,sh[3]//2],'recseg{}_mfig_x.png'.format(i),cmap='gray')

rec = np.sum(rec*mk,axis=0)    
save_fig(rec[sh[1]//2],'recfin_mfig_z.png'.format(i),cmap='gray')
save_fig(rec[:,sh[2]//2],'recfin_mfig_y.png'.format(i),cmap='gray')
save_fig(rec[:,:,sh[3]//2],'recfin_mfig_x.png'.format(i),cmap='gray')
