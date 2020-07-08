import numpy as np
import os
import glob
import csv

test_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/test_mx/'
log_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_005_devrr0_1_roir0/test_mx/'
achans = [4,8,16]

samples = []
with open(os.path.join(test_dir,'subjects.txt'),'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row)==1
        samples.append(row[0])

opts_labels = {'lin':'Lin','nneigh':'NNbor','svm':'SVM','mlp':'MLP'}
score_acc = {opt:[] for opt in opts_labels.keys()}
score_f1 = {opt:[] for opt in opts_labels.keys()}

fieldnames = ['ML method','lin','svm','nneigh','mlp']
with open(os.path.join(test_dir,'fs_gen_summ.csv'),'r') as csv_file:
    reader = csv.DictReader(csv_file,fieldnames=fieldnames)
    for row in reader:
        for opt in opts_labels.keys():
            row_label = row['ML method']
            if row_label == 'score acc':
                score_acc[opt].append(float(row[opt]))
            if row_label == 'score f1':
                score_f1[opt].append(float(row[opt]))

for ac in achans:
    ldir = log_dir
    with open(os.path.join(ldir,'aa_all_gen_summ.csv'),'r') as csv_file:
        reader = csv.DictReader(csv_file,fieldnames=fieldnames)
        for row in reader:
            for opt in opts_labels.keys():
                row_label = row['ML method']
                if row_label == 'score acc':
                    score_acc[opt].append(float(row[opt]))
                if row_label == 'score f1':
                    score_f1[opt].append(float(row[opt]))

print(''.join([' & ','& FS ']+['& AA{} '.format(ac) for ac in achans]+['\\\\\\hline']))
for opt in opts_labels.keys():
    val = max(score_acc[opt])
    text = ' {} & Acc '.format(opts_labels[opt])
    for acc in score_acc[opt]:
        if acc == val:
            text += ' & \\textbf{{{:.2f}}} '.format(acc)
        else:
            text += ' & {:.2f} '.format(acc)
    text += '\\\\\\hline'
    print(text)
    
    val = max(score_f1[opt])
    text = ' {} & F1 '.format(opts_labels[opt])
    for f1 in score_f1[opt]:
        if f1 == val:
            text += ' & \\textbf{{{:.2f}}} '.format(f1)
        else:
            text += ' & {:.2f} '.format(f1)
    text += '\\\\\\hline'
    print(text)

