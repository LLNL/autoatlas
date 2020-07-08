import numpy as np
import os
import glob
import csv
import nibabel as nib
import matplotlib.pyplot as plt

test_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/test_mx/'
log_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_005_devrr0_1_devrm0_9{}/test_mx/'
suffixes = ['','_emb8','_emb16']
achans = [4,8,16]
num_labels = 16
opts_labels = {'lin':'Lin','nneigh':'NNbor','svm':'SVM','mlp':'MLP'}
subjs = ['131924']
num_labels = 16

def save_fig(arr,filen,cmap=None,out_dir='./'):
    plt.imshow(arr,cmap=cmap)
    plt.axis('off')
    #if len(arr.shape)<3:
    #    plt.colorbar()
    plt.savefig(os.path.join(out_dir,filen),bbox_inches='tight')
    plt.close()

fieldnames = ['ML method','lin','svm','nneigh','mlp']
for idx,ac in enumerate(achans):
    for opt in fieldnames[1:]:
        ldir = log_dir.format(suffixes[idx])
        fv_imp = np.zeros(num_labels,dtype=float)
        with open(os.path.join(ldir,'aa_all_strun_summ.csv'),'r') as csv_file:
            reader = csv.DictReader(csv_file,fieldnames=fieldnames)
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
                
            score[mask==0.0] = 0
            rank[mask==0.0] = 0
            perc_75[mask==0.0] = 0
            perc_50[mask==0.0] = 0
            T1[mask==0.0] = 0

            sh = score.shape
            out_folder = os.path.join(os.path.join('figs','rrank_{}_{}'.format(ac,opt)),sub)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder,exist_ok=True)

            score = np.transpose(score,axes=(2,0,1))[::-1]
            save_fig(score[sh[0]//2],'score_z.png',out_dir=out_folder)
            save_fig(score[:,sh[1]//2],'score_y.png',out_dir=out_folder)
            save_fig(score[:,:,sh[2]//2],'score_x.png',out_dir=out_folder)

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
            
