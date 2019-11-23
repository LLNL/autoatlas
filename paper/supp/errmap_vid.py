import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from skvideo.io import vwrite
viridis = cm.get_cmap('viridis_r',128)

#log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc32_11_labels16_smooth0.1_devr1.0_freqs0.05'
achan = 4
log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05'.format(achan)
optimizers = ['lin']
aa_prefix = '_aa{}_tag{}_opt{}.npz'
fs_prefix = '_fs_tag{}_opt{}.npz'
aa_0 = 3
aa_step = 3
max_vols = 12
num_rows = 2
num_cols = max_vols//num_rows
colorbar_scale = 'lin'
FPS = 4

test_folder = os.path.join(log_dir.format(achan),'test_aa')
test_files = [f for f in os.listdir(test_folder) if f[-3:]=='.h5']
test_files.sort()

column_tags = ['Strength_Unadj']
#column_tags = ['Strength_Unadj','Strength_AgeAdj','PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','IWRD_TOT','ListSort_Unadj','ER40_CR','LifeSatisf_Unadj','Endurance_Unadj','Dexterity_Unadj']

for tag in column_tags:
    out_folder = os.path.join(log_dir,'supp','errm','tag{}'.format(tag))
    if not os.path.exists(out_folder):
        os.makedirs(out_folder) 

    for i,opt in enumerate(optimizers): 
        pred_folder= os.path.join(log_dir,'pred_aa')
        aa_pred_file = os.path.join(pred_folder,'pred'+aa_prefix.format(achan,tag,opt))
        aa_perfm_file = os.path.join(pred_folder,'perfm'+aa_prefix.format(achan,tag,opt))

        aa_pred = np.load(aa_pred_file)['test_pred']
        aa_true = np.load(aa_pred_file)['test_true']
        aa_ids = np.load(aa_pred_file)['test_ids']
        aa_err = np.squeeze(np.absolute(aa_pred[0]-aa_true[np.newaxis]))
        ptile = np.percentile(aa_err,2.5)
        print('Number of samples under 5th percentile is {}'.format(np.sum(aa_err<ptile)))
        aa_pred = aa_pred[aa_0::aa_step]
    
        svol,mvol,gtvol,keys = [],[],[],[]
        num_vol = 0
        for filen in test_files:
            ID = filen.split('_')[0]
            IDmk = aa_ids==ID
            if aa_err[IDmk] < ptile:
                keys.append(aa_err[IDmk])
                with h5py.File(os.path.join(test_folder,filen),'r') as f:
                    gt = np.array(f['ground_truth'])
                    seg = np.array(f['segmentation'])
                    mask = np.array(f['mask']).astype(bool)
                scores = np.squeeze(np.absolute(aa_pred[:,IDmk]-aa_true[IDmk])/aa_true[IDmk])
                svol.append(np.sum(seg*scores[:,np.newaxis,np.newaxis,np.newaxis],axis=0))
                mvol.append(mask)
                gtvol.append(gt)
                num_vol = num_vol+1
            if num_vol>=max_vols:
                break      
 
        svol = [x for _, x in sorted(zip(keys,svol), key=lambda pair: pair[0])]
        mvol = [x for _, x in sorted(zip(keys,mvol), key=lambda pair: pair[0])]
        gtvol = [x for _, x in sorted(zip(keys,gtvol), key=lambda pair: pair[0])]
        newkeys = [k for k in keys] 
        newkeys = [x for _, x in sorted(zip(keys,newkeys), key=lambda pair: pair[0])]

        minval,maxval = 0.0,0.1 
        for j in range(max_vols):
            svol[j][np.bitwise_not(mvol[j])] = maxval
            svol[j][svol[j]>maxval] = maxval       
            svol[j] = (viridis(svol[j]/maxval)*255).astype(np.uint8)

        frames_x = np.zeros((96,96*2*num_rows,96*num_cols,4),dtype=np.uint8)
        frames_y = np.zeros((96,96*2*num_rows,96*num_cols,4),dtype=np.uint8)
        frames_z = np.zeros((96,96*2*num_rows,96*num_cols,4),dtype=np.uint8)
        for j in range(max_vols):
            print(newkeys[j])
            col = j%num_cols
            row = j//num_cols    
            gt = np.stack((gtvol[j],gtvol[j],gtvol[j],gtvol[j]),axis=-1)
            gt = ((gt/np.max(gt))*255).astype(np.uint8)
            frames_x[:,row*96:(row+1)*96,col*96:(col+1)*96] = np.transpose(gt,(2,0,1,3)) 
            frames_y[:,row*96:(row+1)*96,col*96:(col+1)*96] = np.transpose(gt,(1,0,2,3)) 
            frames_z[:,row*96:(row+1)*96,col*96:(col+1)*96] = gt
            frames_x[:,(row+num_rows)*96:(row+num_rows+1)*96,col*96:(col+1)*96] = np.transpose(svol[j],(2,0,1,3)) 
            frames_y[:,(row+num_rows)*96:(row+num_rows+1)*96,col*96:(col+1)*96] = np.transpose(svol[j],(1,0,2,3)) 
            frames_z[:,(row+num_rows)*96:(row+num_rows+1)*96,col*96:(col+1)*96] = svol[j]
        
        vwrite(os.path.join(out_folder,'olay-horz.mp4'.format(tag,opt,colorbar_scale)),frames_x,backend='ffmpeg',inputdict={'-r':'{}'.format(FPS)},outputdict={'-r':'{}'.format(FPS),'-crf':str(17),'-vcodec':'libx264','-pix_fmt':'yuv420p','-preset':'ultrafast'})
        vwrite(os.path.join(out_folder,'olay-cor.mp4'.format(tag,opt,colorbar_scale)),frames_y,backend='ffmpeg',inputdict={'-r':'{}'.format(FPS)},outputdict={'-r':'{}'.format(FPS),'-crf':str(17),'-vcodec':'libx264','-pix_fmt':'yuv420p','-preset':'ultrafast'})
        vwrite(os.path.join(out_folder,'olay-sag.mp4'.format(tag,opt,colorbar_scale)),frames_z,backend='ffmpeg',inputdict={'-r':'{}'.format(FPS)},outputdict={'-r':'{}'.format(FPS),'-crf':str(17),'-vcodec':'libx264','-pix_fmt':'yuv420p','-preset':'ultrafast'})

