import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

achans = [4,16]

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05'
optimizers = ['lin','nneigh','boost','mlp']
opts_labels = ['Linear','NNeighbor','Boosting','MLPerceptron']
aa_prefix = '_aa{}_tag{}_opt{}.npz'
fs_prefix = '_fs_tag{}_opt{}.npz'
aa_0 = 5
aa_step = 3

column_tags = ['Strength_Unadj','Gender']
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

bar_width = 1.0/(len(optimizers)+1)
num_bars = 16
num_opts = len(optimizers)
pos = np.arange(num_bars)
bar_ticks = ['L{}'.format(i) for i in range(num_bars)]

for ac in achans: 
    for tag in column_tags:
        out_folder = os.path.join(log_dir.format(ac),'paper','pred','tag{}'.format(tag))
        if not os.path.exists(out_folder):
            os.makedirs(out_folder) 

        score0 = np.zeros((num_opts,num_bars),dtype=float)
        score1 = np.zeros((num_opts,num_bars),dtype=float)
        for i,opt in enumerate(optimizers): 
            pred_folder= os.path.join(log_dir.format(ac),'pred_aa')
            aa_perfm_file = os.path.join(pred_folder,'perfm'+aa_prefix.format(ac,tag,opt))
            aa_pred_file = os.path.join(pred_folder,'pred'+aa_prefix.format(ac,tag,opt))
            fs_perfm_file = os.path.join(pred_folder,'perfm'+fs_prefix.format(tag,opt))

            aa_perfm = np.load(aa_perfm_file)
            score0[i] = aa_perfm['test_perfs'][aa_0::aa_step,0]
            score1[i] = aa_perfm['test_perfs'][aa_0::aa_step,1]

            aa_pred = np.load(aa_pred_file)['test_pred'][aa_0::aa_step]
            aa_true = np.load(aa_pred_file)['test_true']

            idx = np.nonzero(score0==np.max(score0))
            idx = idx[0][0]
            yvals = aa_pred[idx,:]
            plt.scatter(aa_true,yvals)
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.title('Label {}'.format(idx-1)) #-1 because first one includes all embeddings
            plt.savefig(os.path.join(out_folder,'scat_highscore0_aenc{}_tag{}_opt{}.png'.format(ac,tag,opt)),bbox_inches='tight')
            plt.close()
            
            idx = np.nonzero(score0==np.min(score0))
            idx = idx[0][0]
            yvals = aa_pred[idx,:]
            plt.scatter(aa_true,yvals)
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.title('Label {}'.format(idx-1)) #-1 because first one includes all embeddings
            plt.savefig(os.path.join(out_folder,'scat_lowscore0_aenc{}_tag{}_opt{}.png'.format(ac,tag,opt)),bbox_inches='tight')
            plt.close()

        off = bar_width*len(optimizers)/2
     
        plt.figure(figsize=(15,3))
        rects = []
        for i in range(num_opts):    
            rects.append(plt.bar(pos+i*bar_width,score0[i],bar_width))
        plt.xticks(pos,bar_ticks)
        plt.legend(rects,opts_labels,loc='upper center',ncol=num_opts,bbox_to_anchor=(0.5,1.2),fancybox=True,shadow=True)
        plt.savefig(os.path.join(out_folder,'bar_score0_aenc{}_tag{}.png'.format(ac,tag,opt)),bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(15,3))
        score1[score1<0] = 0
        rects = []
        for i in range(num_opts):    
            rects.append(plt.bar(pos+i*bar_width,score1[i],bar_width))
        plt.xticks(pos,bar_ticks)
        plt.legend(rects,opts_labels,loc='upper center',ncol=num_opts,bbox_to_anchor=(0.5,1.2),fancybox=True,shadow=True)
        plt.savefig(os.path.join(out_folder,'bar_score1_aenc{}_tag{}.png'.format(ac,tag,opt)),bbox_inches='tight')
        plt.close()


