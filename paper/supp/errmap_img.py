import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext,FuncFormatter
from matplotlib import cm
viridis = cm.get_cmap('viridis',128)

#log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc32_11_labels16_smooth0.1_devr1.0_freqs0.05'
achan = 4
log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05'.format(achan)
optimizers = ['lin']
aa_prefix = '_aa{}_tag{}_opt{}.npz'
fs_prefix = '_fs_tag{}_opt{}.npz'
aa_0 = 3
aa_step = 3
max_vols = 12
colorbar_scale = 'lin'

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

def logfmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def save_fig(arr,filen,cmap=None,vmin=None,vmax=None,cbar=False):
    plt.figure()
    if colorbar_scale == 'log':
        plt.imshow(arr,cmap=cmap,norm=LogNorm(),vmin=vmin,vmax=vmax)
    else:
        plt.imshow(arr,cmap=cmap,vmin=vmin,vmax=vmax)
    plt.axis('off')
    if cbar:
        #if colorbar_scale == 'log':
        #    plt.colorbar(format=FuncFormatter(logfmt))
        #else:     
        plt.colorbar()
    plt.savefig(filen,bbox_inches='tight')
    plt.close()

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
        #ptile = np.percentile(aa_err,1.3)
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
                #print(np.min(scores),np.max(scores))
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
 
        cmap = mpl.cm.get_cmap('viridis_r')
        minval,maxval = 0.0,0.1
        for j in range(max_vols):
            print(newkeys[j])
            sh = svol[j].shape
            svol[j][svol[j]>maxval] = maxval       
            svol[j][np.bitwise_not(mvol[j])] = maxval
            cbar = True if j==0 else False
            save_fig(gtvol[j][:,:,sh[-1]//2],os.path.join(out_folder,'gt{}_x.png'.format(j)),cmap='gray')
            save_fig(gtvol[j][:,sh[-2]//2],os.path.join(out_folder,'gt{}_y.png'.format(j)),cmap='gray')
            save_fig(gtvol[j][sh[-3]//2],os.path.join(out_folder,'gt{}_z.png'.format(j)),cmap='gray')
            save_fig(svol[j][:,:,sh[-1]//2],os.path.join(out_folder,'olay{}_tag{}_opt{}_cbar{}_x.png'.format(j,tag,opt,colorbar_scale)),cmap=cmap,vmin=minval,vmax=maxval,cbar=cbar)
            save_fig(svol[j][:,sh[-2]//2],os.path.join(out_folder,'olay{}_tag{}_opt{}_cbar{}_y.png'.format(j,tag,opt,colorbar_scale)),cmap=cmap,vmin=minval,vmax=maxval,cbar=cbar)
            save_fig(svol[j][sh[-3]//2],os.path.join(out_folder,'olay{}_tag{}_opt{}_cbar{}_z.png'.format(j,tag,opt,colorbar_scale)),cmap=cmap,vmin=minval,vmax=maxval,cbar=cbar)
            

