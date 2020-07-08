import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import csv
import nibabel as nib

tags = ['mxvol_labs16_smooth0_005_devrr0_1_devrm0_9','mxvol_labs16_smooth0_005_devrr0_1_devrm0_9_emb16']
save_dirs = ['emb4','emb16']
num_labels = 16

smpl_list = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/test_mx/subjects.txt'

def read_csv(filen):
    with open(filen,mode='r') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        data = []
        for i,row in enumerate(csv_reader):
            if i!=0:
                data.append(np.array(row[1:],dtype=float))
    return np.stack(data,axis=0)

samples = []
with open(smpl_list,'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row)==1
        samples.append(row[0])

means,stds,row = [],[],''
for tag in tags:
    print('Processing tag {}'.format(tag))
    data_mat = []
    out_folder = os.path.join('figs',tag)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder) 
    
    for sub in samples:
        log_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/{}/test_mx/{}/'.format(tag,sub)
        mask_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/test_mx/{}/'.format(sub)

        data = read_csv(os.path.join(log_dir,'olap_nnone.csv'))
        assert data.shape[1] == 4
        mask = nib.load(os.path.join(mask_dir,'mask.nii.gz')).get_fdata()
        mask_sum = np.sum(mask)
        data_mat.append(data/mask_sum*100)
        
    data_mat = np.stack(data_mat,axis=0)
    means.append(np.mean(data_mat,axis=0))
    stds.append(np.std(data_mat,axis=0))
    row += '& TT1 & TT2 & TT3 & TT4 '
print(row+'\\\\\\hline')

for i in range(num_labels):
    row = 'AA{}'.format(i)
    for tidx in range(len(tags)):
        for j in range(means[tidx].shape[1]):
            if means[tidx][i,j] > 1:
                row += ' & \\textbf{{{:.2f} ({:.2f})}}'.format(means[tidx][i,j],stds[tidx][i,j])
            else:
                row += ' & {:.2f} ({:.2f})'.format(means[tidx][i,j],stds[tidx][i,j])
    print(row+'\\\\\\hline')

