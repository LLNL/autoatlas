import numpy as np
import os
import h5py
from skvideo.io import vwrite
import matplotlib.pyplot as plt

achan = 4
test_start = 0
num_test = 24

num_rows = int(3)
num_cols = int(num_test/num_rows)

width = num_cols*96
height = 2*num_rows*96
num_frames = 96 
FPS = 4

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05'
num_labels = 16

rgb_list = np.array([[128,0,0],[170,110,40],[128,128,0],[0,128,128],[0,0,128],[255,255,255],[230,25,75],[245,130,48],[255,255,25],[210,245,60],[60,180,75],[70,240,240],[0,130,200],[145,30,180],[240,50,230],[128,128,128]]).astype(np.uint8)[:num_labels]

out_folder = os.path.join(log_dir.format(achan),'supp','seg')
if not os.path.exists(out_folder):
    os.makedirs(out_folder) 

test_folder = os.path.join(log_dir.format(achan),'test_aa')
test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-3:]=='.h5']
test_files.sort()

frames_x = np.zeros((num_frames,height,width,3),dtype=np.uint8)
frames_y = np.zeros((num_frames,height,width,3),dtype=np.uint8)
frames_z = np.zeros((num_frames,height,width,3),dtype=np.uint8)
for i in range(test_start,test_start+num_test):
    col = (i-test_start)%num_cols
    row = (i-test_start)//num_cols

    print(test_files[i])
    with h5py.File(test_files[i],'r') as f:
        gt = np.array(f['ground_truth'])
        seg = np.array(f['segmentation'])
        mk = np.array(f['mask'])
    sh = gt.shape
    seg = seg*mk[np.newaxis]
    seg = np.sum(seg[:,:,:,:,np.newaxis]*rgb_list[:,np.newaxis,np.newaxis,np.newaxis],axis=0).astype(np.uint8)

    sh = gt.shape
    gt = np.stack((gt,gt,gt),axis=-1)
    gt = (255*gt/np.max(gt)).astype(np.uint8)
    frames_x[:,row*sh[0]:(row+1)*sh[0],col*sh[1]:(col+1)*sh[1]] = np.transpose(gt,(2,0,1,3)) 
    frames_x[:,(row+num_rows)*sh[0]:(row+num_rows+1)*sh[0],col*sh[1]:(col+1)*sh[1]] = np.transpose(seg,(2,0,1,3))
    frames_y[:,row*sh[0]:(row+1)*sh[0],col*sh[1]:(col+1)*sh[1]] = np.transpose(gt,(1,0,2,3)) 
    frames_y[:,(row+num_rows)*sh[0]:(row+num_rows+1)*sh[0],col*sh[1]:(col+1)*sh[1]] = np.transpose(seg,(1,0,2,3))
    frames_z[:,row*sh[0]:(row+1)*sh[0],col*sh[1]:(col+1)*sh[1]] = gt
    frames_z[:,(row+num_rows)*sh[0]:(row+num_rows+1)*sh[0],col*sh[1]:(col+1)*sh[1]] = seg

#plt.imshow(frames[96//2])
#plt.show()

vwrite(os.path.join(out_folder,'seg-horz.mp4'),frames_x,backend='ffmpeg',inputdict={'-r':'{}'.format(FPS)},outputdict={'-r':'{}'.format(FPS),'-crf':str(17),'-vcodec':'libx264','-pix_fmt':'yuv420p','-preset':'ultrafast'})
vwrite(os.path.join(out_folder,'seg-cor.mp4'),frames_y,backend='ffmpeg',inputdict={'-r':'{}'.format(FPS)},outputdict={'-r':'{}'.format(FPS),'-crf':str(17),'-vcodec':'libx264','-pix_fmt':'yuv420p','-preset':'ultrafast'})
vwrite(os.path.join(out_folder,'seg-sag.mp4'),frames_z,backend='ffmpeg',inputdict={'-r':'{}'.format(FPS)},outputdict={'-r':'{}'.format(FPS),'-crf':str(17),'-vcodec':'libx264','-pix_fmt':'yuv420p','-preset':'ultrafast'})

