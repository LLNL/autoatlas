import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc32_11_labels16_smooth0.1_devr1.0_freqs0.05'
achan = 4
log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05'.format(achan)
optimizers = ['lin']
aa_prefix = '_aa{}_tag{}_opt{}.npz'
fs_prefix = '_fs_tag{}_opt{}.npz'
aa_0 = 3
aa_step = 3
max_vols = 6

test_folder = os.path.join(log_dir.format(achan),'test_aa')
test_files = [f for f in os.listdir(test_folder) if f[-3:]=='.h5']
test_files.sort()

column_tags = ['Strength_Unadj']
#column_tags = ['Strength_Unadj','Strength_AgeAdj','PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','IWRD_TOT','ListSort_Unadj','ER40_CR','LifeSatisf_Unadj','Endurance_Unadj','Dexterity_Unadj']

FONTSZ = 12
LINEIND_WIDTH = 3.0
plt.rc('font', size=FONTSZ)          # controls default text sizes
plt.rc('axes', titlesize=FONTSZ)     # fontsize of the axes title
plt.rc('axes', labelsize=FONTSZ)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONTSZ)    # legend fontsize
plt.rc('figure', titlesize=FONTSZ)  # fontsize of the figure title

def save_fig(arr,filen,cmap=None,vmin=0,vmax=1,cbar=False):
    plt.figure()
    plt.imshow(arr,cmap=cmap,vmin=0)
    plt.axis('off')
    if cbar:
        plt.colorbar()
    plt.savefig(filen,bbox_inches='tight')
    plt.close()
    print('vmin={},vmax={}'.format(vmin,vmax))

for tag in column_tags:
    out_folder = os.path.join(log_dir,'paper','pred','tag{}'.format(tag))
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
        ptile = np.percentile(aa_err,1.3)
        print('Number of samples under 5th percentile is {}'.format(np.sum(aa_err<ptile)))
        aa_pred = aa_pred[aa_0::aa_step]
    
        svol = np.zeros((max_vols,96,96,96),dtype=float)
        mvol = np.zeros((max_vols,96,96,96),dtype=bool)
        num_vol = 0
        for filen in test_files:
            ID = filen.split('_')[0]
            IDmk = aa_ids==ID
            if aa_err[IDmk] < ptile:
                with h5py.File(os.path.join(test_folder,filen),'r') as f:
                    gt = np.array(f['ground_truth'])
                    seg = np.array(f['segmentation'])
                    mask = np.array(f['mask']).astype(bool)
                scores = np.squeeze(np.absolute(aa_pred[:,IDmk]-aa_true[IDmk])/aa_true[IDmk])
                print(np.min(scores),np.max(scores))
                svol[num_vol] = np.sum(seg*scores[:,np.newaxis,np.newaxis,np.newaxis],axis=0)
                mvol[num_vol] = mask
                num_vol = num_vol+1
            if num_vol>=max_vols:
                break      
 
        maxval = np.max(svol) 
        minval = np.min(svol) 
        cmap = mpl.cm.get_cmap('viridis_r')
#            save_fig(svol[sh[0]//2],os.path.join(out_folder,'olay_r2_tag{}_opt{}_z.png'.format(tag,opt)),cmap=cmap)
#            save_fig(svol[:,sh[1]//2],os.path.join(out_folder,'olay_r2_tag{}_opt{}_y.png'.format(tag,opt)),cmap=cmap)
        for j in range(max_vols):
            svol[j][np.bitwise_not(mvol[j])] = maxval
            cbar = True if j==0 else False
            save_fig(svol[j,:,:,svol.shape[-1]//2],os.path.join(out_folder,'olay{}_tag{}_opt{}_x.png'.format(j,tag,opt)),cmap=cmap,vmin=minval,vmax=maxval,cbar=cbar)
            

