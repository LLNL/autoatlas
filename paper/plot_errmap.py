import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc32_11_labels16_smooth0.1_devr1.0_freqs0.05'
log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05'
optimizers = ['lin']
aa_prefix = '_aa4_tag{}_opt{}.npz'
fs_prefix = '_fs_tag{}_opt{}.npz'
aa_0 = 5
aa_step = 3
max_vols = 20

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

def save_fig(arr,filen,cmap=None,vmin=0,vmax=1):
    plt.figure()
    plt.imshow(arr,cmap=cmap,vmin=0)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(filen,bbox_inches='tight')
    plt.close()

for tag in column_tags:
    out_folder = os.path.join(log_dir,'paper','pred','tag{}'.format(tag))
    if not os.path.exists(out_folder):
        os.makedirs(out_folder) 

    for i,opt in enumerate(optimizers): 
        pred_folder= os.path.join(log_dir,'pred_aa')
        aa_pred_file = os.path.join(pred_folder,'pred'+aa_prefix.format(tag,opt))
        fs_perfm_file = os.path.join(pred_folder,'perfm'+fs_prefix.format(tag,opt))

        aa_pred = np.load(aa_pred_file)['test_pred'][aa_0::aa_step]
        aa_true = np.load(aa_pred_file)['test_true']
        aa_ids = np.load(aa_pred_file)['test_ids']

        svol = np.zeros((max_vols,96,96,96),dtype=float)
        mvol = np.zeros((max_vols,96,96,96),dtype=bool)
        for j in range(max_vols):
            ID = aa_ids[j]
            with h5py.File(os.path.join(log_dir,'test_aa','{}_T1w_brain_2_aa.h5'.format(ID)),'r') as f:
                gt = np.array(f['ground_truth'])
                seg = np.array(f['segmentation'])
                mask = np.array(f['mask']).astype(bool)
            scores = np.absolute(aa_pred[:,j]-aa_true[j])/aa_true[j]
            print(scores)
            svol[j] = np.sum(seg*scores[:,np.newaxis,np.newaxis,np.newaxis],axis=0)
            mvol[j] = mask      
 
        maxval = np.max(svol) 
        cmap = mpl.cm.get_cmap('viridis_r')
#            save_fig(svol[sh[0]//2],os.path.join(out_folder,'olay_r2_tag{}_opt{}_z.png'.format(tag,opt)),cmap=cmap)
#            save_fig(svol[:,sh[1]//2],os.path.join(out_folder,'olay_r2_tag{}_opt{}_y.png'.format(tag,opt)),cmap=cmap)
        for j in range(max_vols):
            svol[j][np.bitwise_not(mvol[j])] = maxval
            save_fig(svol[j,:,:,svol.shape[-1]//2],os.path.join(out_folder,'olay{}_tag{}_opt{}_x.png'.format(j,tag,opt)),cmap=cmap,vmin=0,vmax=maxval)
            

