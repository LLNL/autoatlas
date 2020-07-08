import numpy as np
import os
import glob
import csv

test_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/test_mx/'
log_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_005_devrr0_1_devrm0_9{}/test_mx/'
suffixes = ['','_emb8','_emb16']
achans = [4,8,16]
num_labels = 16
opts_labels = {'lin':'Lin','nneigh':'NNbor','svm':'SVM','mlp':'MLP'}

samples = []
with open(os.path.join(test_dir,'subjects.txt'),'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row)==1
        samples.append(row[0])

print(''.join([' $C_a$ ','& Optim ']+['& $f_{}$ '.format(lab) for lab in range(num_labels)]+['\\\\\\hline']))
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
        text = ' {} & {} '.format(ac,opts_labels[opt])
        thresh = np.percentile(fv_imp,75)
        for lab in range(num_labels):
            if fv_imp[lab] > thresh:
                text += '& \\textbf{{{:.2f}}} '.format(fv_imp[lab])
            else:
                text += '& {:.2f} '.format(fv_imp[lab])
        print(text+'\\\\\\hline')

