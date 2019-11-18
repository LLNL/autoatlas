import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc32_11_labels16_smooth0.1_devr1.0_freqs0.05'
log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05'
optimizers = ['lin','nneigh','lasso','mlp']
aa_prefix = '_aa4_tag{}_opt{}.npz'
fs_prefix = '_fs_tag{}_opt{}.npz'
aa_0 = 2
aa_step = 3

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

test_folder = os.path.join(log_dir,'test_aa')
test_files = [os.path.join(test_folder,f) for f in os.listdir(test_folder) if f[-3:]=='.h5']
    
with h5py.File(test_files[0],'r') as f:
    gt = np.array(f['ground_truth'])
    seg = np.array(f['segmentation'])
    mask = np.array(f['mask'])[np.newaxis]

def save_fig(arr,filen,cmap=None):
    plt.figure()
    plt.imshow(arr,cmap=cmap)
    plt.axis('off')
    plt.colorbar()
    plt.savefig(filen,bbox_inches='tight')
    plt.close()

bar_width = 1.0/(len(optimizers)+1)
num_bars = 18
num_opts = len(optimizers)
pos = np.arange(num_bars)
bar_ticks = ['F','A']+['{}'.format(i) for i in range(16)]
 
for tag in column_tags:
    out_folder = os.path.join(log_dir,'paper','pred','tag{}'.format(tag))
    if not os.path.exists(out_folder):
        os.makedirs(out_folder) 

    mse = np.zeros((num_opts,num_bars),dtype=float)
    r2 = np.zeros((num_opts,num_bars),dtype=float)
    for i,opt in enumerate(optimizers): 
        pred_folder= os.path.join(log_dir,'pred_aa')
        aa_perfm_file = os.path.join(pred_folder,'perfm'+aa_prefix.format(tag,opt))
        aa_pred_file = os.path.join(pred_folder,'pred'+aa_prefix.format(tag,opt))
        fs_perfm_file = os.path.join(pred_folder,'perfm'+fs_prefix.format(tag,opt))

        aa_perfm = np.load(aa_perfm_file)
        aa_mse = aa_perfm['test_perfs'][aa_0::aa_step,0]
        aa_r2 = aa_perfm['test_perfs'][aa_0::aa_step,1]

        aa_pred = np.load(aa_pred_file)['test_pred'][aa_0::aa_step]
        aa_true = np.load(aa_pred_file)['test_true']

        fs_perfm = np.load(fs_perfm_file)
        fs_mse = fs_perfm['test_perfs'][0,0]
        fs_r2 = fs_perfm['test_perfs'][0,1]

        mse[i,0] = fs_mse
        mse[i,1:] = aa_mse
        r2[i,0] = fs_r2
        r2[i,1:] = aa_r2
        
        scores = np.array(aa_r2[1:])
        scores[scores<0] = 0
        svol = np.sum(seg*scores[:,np.newaxis,np.newaxis,np.newaxis],axis=0)
        sh = svol.shape
        cmap = mpl.cm.get_cmap('viridis')
        save_fig(svol[sh[0]//2],os.path.join(out_folder,'olay_r2_tag{}_opt{}_z.png'.format(tag,opt)),cmap=cmap)
        save_fig(svol[:,sh[1]//2],os.path.join(out_folder,'olay_r2_tag{}_opt{}_y.png'.format(tag,opt)),cmap=cmap)
        save_fig(svol[:,:,sh[2]//2],os.path.join(out_folder,'olay_r2_tag{}_opt{}_x.png'.format(tag,opt)),cmap=cmap)
        
        scores = np.array(aa_mse[1:])
        scores[scores<0] = 0
        svol = np.sum(seg*scores[:,np.newaxis,np.newaxis,np.newaxis],axis=0)
        sh = svol.shape
        cmap = mpl.cm.get_cmap('viridis_r')
        save_fig(svol[sh[0]//2],os.path.join(out_folder,'olay_mse_tag{}_opt{}_z.png'.format(tag,opt)),cmap=cmap)
        save_fig(svol[:,sh[1]//2],os.path.join(out_folder,'olay_mse_tag{}_opt{}_y.png'.format(tag,opt)),cmap=cmap)
        save_fig(svol[:,:,sh[2]//2],os.path.join(out_folder,'olay_mse_tag{}_opt{}_x.png'.format(tag,opt)),cmap=cmap)

        idx = np.nonzero(aa_mse==max(aa_mse[1:]))
        idx = idx[0][0]
        yvals = aa_pred[idx,:]
        plt.scatter(aa_true,yvals)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title('Label {}'.format(idx-1)) #-1 because first one includes all embeddings
        plt.savefig(os.path.join(out_folder,'scat_highmse_tag{}_opt{}.png'.format(tag,opt)),bbox_inches='tight')
        plt.close()
        
        idx = np.nonzero(aa_mse==min(aa_mse[1:]))
        idx = idx[0][0]
        yvals = aa_pred[idx,:]
        plt.scatter(aa_true,yvals)
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title('Label {}'.format(idx-1)) #-1 because first one includes all embeddings
        plt.savefig(os.path.join(out_folder,'scat_lowmse_tag{}_opt{}.png'.format(tag,opt)),bbox_inches='tight')
        plt.close()

    off = bar_width*len(optimizers)/2
 
    plt.figure(figsize=(8,3))
    rects = []
    for i in range(num_opts):    
        rects.append(plt.bar(pos+i*bar_width,mse[i],bar_width))
    plt.xticks(pos,bar_ticks)
    plt.ylabel('NRMSE')
    plt.legend(rects,optimizers,loc='lower center',ncol=num_opts)
    plt.savefig(os.path.join(out_folder,'bar_nrmse_tag{}.png'.format(tag,opt)),bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8,3))
    rects = []
    for i in range(num_opts):    
        r2temp = np.copy(r2[i])
        r2temp[r2temp<-0.5]=-0.5
        rects.append(plt.bar(pos+i*bar_width,r2temp,bar_width))
    plt.xticks(pos,bar_ticks)
    plt.ylabel('R2 score')
    plt.legend(rects,optimizers,loc='lower center',ncol=num_opts)
    plt.savefig(os.path.join(out_folder,'bar_r2_tag{}.png'.format(tag,opt)),bbox_inches='tight')
    plt.close()


