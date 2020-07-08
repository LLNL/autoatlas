import numpy as np
import csv
import os
import nibabel as nib
#from scipy.stats import entropy as scipy_entropy

tag_list = ['mxvol_labs16_relr0_0_smooth0_005_devrr0_1_devrm0_9','mxvol_labs16_smooth0_0_devrr0_1_devrm0_9','mxvol_labs16_smooth0_005_devrr0_0_devrm0_9','mxvol_labs16_smooth0_005_devrr0_1_devrm0_9']
rel_reg = [0.0,1.0,1.0,1.0]
smooth_reg = [0.005,0.0,0.005,0.005]
devr_reg = [0.1,0.1,0.0,0.1]

smpl_list = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/test_mx/subjects.txt'

fieldnames = ['ML method','lin','svm','nneigh','mlp']
opts_labels = {'lin':'Lin','nneigh':'NNbor','svm':'SVM','mlp':'MLP'}
r2 = {opt:[] for opt in fieldnames}
all_entropy,all_prob01 = [],[]
for tag in tag_list:
    log_dir = os.path.join(os.path.join('/p/gpfs1/mohan3/Data/TBI/HCP/2mm/',tag),'test_mx')
    data_dir = os.path.join('/p/gpfs1/mohan3/Data/TBI/HCP/2mm/','test_mx')

    with open(os.path.join(log_dir,'aa_all_strun_summ.csv'),'r') as csv_file:  
        reader = csv.DictReader(csv_file,fieldnames=fieldnames)
        for row in reader:
            row_label = row['ML method']
            if row_label == 'score r2':
                for opt in fieldnames[1:]:
                    r2[opt].append(float(row[opt]))   

text = '$\\lambda_{RE}$ & $\\lambda_{NSS}$ & $\\lambda_{AD}$'
text += ''.join([' & {}'.format(opts_labels[opt]) for opt in fieldnames[1:]])
text += '\\\\\\hline'
print(text) 
for i in range(len(devr_reg)):
    text = '{} & {} & {}'.format(rel_reg[i],smooth_reg[i],devr_reg[i])
    for opt in fieldnames[1:]:
        text += ' & {:.2f}'.format(r2[opt][i])
    text += '\\\\\\hline'
    print(text) 
 
