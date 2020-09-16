import numpy as np
import os
import glob
import csv
import colorcet as cc
import nibabel as nib
import matplotlib.pyplot as plt

test_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/test_mx/'
log_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_005_devrr0_1_devrm0_9{}/test_mx/'
suffixes = ['_emb16']
achans = [16]
num_labels = 16
opts_labels = {'lin':'Lin','nneigh':'NNbor','svm':'SVM','mlp':'MLP'}
subjs = ['131924']
num_labels = 16
rgb_list = np.array([[128,128,128],[70,240,240],[255,255,255],[230,25,75],[0,0,128],[128,128,0],[0,128,128],[170,110,40],[245,130,48],[255,255,25],[128,0,0],[210,245,60],[60,180,75],[0,130,200],[145,30,180],[240,50,230]]).astype(np.uint8)[:num_labels]

FONTSZ = 20
LINEIND_WIDTH = 3.0
plt.rc('font', size=FONTSZ)          # controls default text sizes
plt.rc('axes', titlesize=FONTSZ)     # fontsize of the axes title
plt.rc('axes', labelsize=FONTSZ)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONTSZ)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONTSZ)    # legend fontsize
plt.rc('figure', titlesize=FONTSZ)  # fontsize of the figure title

def save_fig(arr,filen,cmap=None,out_dir='./',no_cbar=False):
    plt.imshow(arr,cmap=cmap,vmin=0)
    plt.axis('off')
    if not no_cbar:
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,filen),bbox_inches='tight')
    plt.close()

fieldnames = ['ML method','lin','svm','nneigh','mlp']
for idx,ac in enumerate(achans):
    for opt in fieldnames[1:]:
        ldir = log_dir.format(suffixes[idx])
        fv_imp = np.zeros(num_labels,dtype=float)
        with open(os.path.join(ldir,'aa_emb_strun_summ.csv'),'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                row_label = row['ML method']
                if 'imp mae' in row_label:
                    lab = int(row_label.split(' ')[0][2:])
                    fv_imp[lab] = float(row[opt])

        temp = fv_imp.argsort()
        fv_rank = np.empty_like(temp)
        fv_rank[temp] = np.arange(1,len(fv_imp)+1)
        fv_p75 = np.percentile(fv_imp,q=75)
        fv_p50 = np.percentile(fv_imp,q=50)
 
        for sub in subjs:
            seg = nib.load(os.path.join(os.path.join(ldir,sub),'seg_vol.nii.gz')).get_fdata()
            T1 = nib.load('/p/gpfs1/mohan3/Data/TBI/HCP/2mm/test_mx/{}/T1.nii.gz'.format(sub)).get_fdata()
            TT5 = nib.load('/p/gpfs1/mohan3/Data/TBI/HCP/2mm/test_mx/{}/5TT.nii.gz'.format(sub)).get_fdata()
            mask = nib.load('/p/gpfs1/mohan3/Data/TBI/HCP/2mm/test_mx/{}/mask.nii.gz'.format(sub)).get_fdata()
            score = np.zeros_like(seg)
            rank = np.zeros_like(seg)
            perc_75 = np.zeros_like(seg)
            perc_50 = np.zeros_like(seg)
            for lab in range(num_labels):
                score[seg==lab] = fv_imp[lab]
                rank[seg==lab] = fv_rank[lab]             
                perc_75[seg==lab] = 1 if fv_imp[lab]>=fv_p75 else 0
                perc_50[seg==lab] = 1 if fv_imp[lab]>=fv_p50 else 0
                
            segrgb = np.zeros((seg.shape[0],seg.shape[1],seg.shape[2],3),dtype=np.uint8)
            TT5rgb = np.zeros((TT5.shape[0],TT5.shape[1],TT5.shape[2],3),dtype=np.uint8)
            for j in range(num_labels):
                segrgb[seg==j] = rgb_list[j]
            segrgb[mask==0.0] = np.array([0,0,0],dtype=np.uint8)
            segrgb = np.transpose(segrgb,axes=(2,0,1,3))[::-1]
            
            for j,lab in enumerate(np.unique(TT5)):
                TT5rgb[TT5==lab] = rgb_list[j]
            TT5rgb[mask==0.0] = np.array([0,0,0],dtype=np.uint8)
            TT5rgb = np.transpose(TT5rgb,axes=(2,0,1,3))[::-1]

            score[mask==0.0] = 0
            rank[mask==0.0] = 0
            perc_75[mask==0.0] = 0
            perc_50[mask==0.0] = 0
            T1[mask==0.0] = 0
            TT5[mask==0.0] = 0

            sh = score.shape
            out_folder = os.path.join(os.path.join('figs','strunrank_{}_{}'.format(ac,opt)),sub)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder,exist_ok=True)

            score = np.transpose(score,axes=(2,0,1))[::-1]
            #save_fig(score[sh[0]//2],'score_z.png',out_dir=out_folder,cmap=cc.cm.rainbow)
            #save_fig(score[:,sh[1]//2],'score_y.png',out_dir=out_folder,cmap=cc.cm.rainbow)
            #save_fig(score[:,:,sh[2]//2],'score_x.png',out_dir=out_folder,cmap=cc.cm.rainbow)
            save_fig(score[sh[0]//2],'score_z.png',out_dir=out_folder,cmap='inferno')
            save_fig(score[:,sh[1]//2],'score_y.png',out_dir=out_folder,cmap='inferno')
            save_fig(score[:,:,sh[2]//2],'score_x.png',out_dir=out_folder,cmap='inferno')

            rank = np.transpose(rank,axes=(2,0,1))[::-1]
            save_fig(rank[sh[0]//2],'rank_z.png',out_dir=out_folder)
            save_fig(rank[:,sh[1]//2],'rank_y.png',out_dir=out_folder)
            save_fig(rank[:,:,sh[2]//2],'rank_x.png',out_dir=out_folder)
            
            perc_75 = np.transpose(perc_75,axes=(2,0,1))[::-1]
            save_fig(perc_75[sh[0]//2],'p75_z.png',out_dir=out_folder)
            save_fig(perc_75[:,sh[1]//2],'p75_y.png',out_dir=out_folder)
            save_fig(perc_75[:,:,sh[2]//2],'p75_x.png',out_dir=out_folder)
            
            perc_50 = np.transpose(perc_50,axes=(2,0,1))[::-1]
            save_fig(perc_50[sh[0]//2],'p50_z.png',out_dir=out_folder)
            save_fig(perc_50[:,sh[1]//2],'p50_y.png',out_dir=out_folder)
            save_fig(perc_50[:,:,sh[2]//2],'p50_x.png',out_dir=out_folder)
            
            T1 = np.transpose(T1,axes=(2,0,1))[::-1]
            save_fig(T1[sh[0]//2],'T1_z.png',out_dir=out_folder)
            save_fig(T1[:,sh[1]//2],'T1_y.png',out_dir=out_folder)
            save_fig(T1[:,:,sh[2]//2],'T1_x.png',out_dir=out_folder)
            
            save_fig(segrgb[sh[0]//2],'seg_z.png',out_dir=out_folder,no_cbar=True)
            save_fig(segrgb[:,sh[1]//2],'seg_y.png',out_dir=out_folder,no_cbar=True)
            save_fig(segrgb[:,:,sh[2]//2],'seg_x.png',out_dir=out_folder,no_cbar=True)
            
            save_fig(TT5rgb[sh[0]//2],'tt5_z.png',out_dir=out_folder,no_cbar=True)
            save_fig(TT5rgb[:,sh[1]//2],'tt5_y.png',out_dir=out_folder,no_cbar=True)
            save_fig(TT5rgb[:,:,sh[2]//2],'tt5_x.png',out_dir=out_folder,no_cbar=True)
            
