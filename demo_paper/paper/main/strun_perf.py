import numpy as np
import os
import glob
import csv

test_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/test_mx/'
log_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_005_devrr0_1_devrm0_9{}/test_mx/'
suffixes = ['','_emb8','_emb16','_emb32','_emb64']
achans = [4,8,16,32,64]

samples = []
with open(os.path.join(test_dir,'subjects.txt'),'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row)==1
        samples.append(row[0])

opts_labels = {'lin':'Lin','nneigh':'NNbor','svm':'SVM','mlp':'MLP'}
score_r2 = {opt:[] for opt in opts_labels.keys()}
score_mae = {opt:[] for opt in opts_labels.keys()}

#fieldnames = ['ML method','lin','svm','nneigh','mlp']
with open(os.path.join(test_dir,'fs_strun_summ.csv'),'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        for opt in opts_labels.keys():
            row_label = row['ML method']
            if row_label == 'score r2':
                score_r2[opt].append(float(row[opt]))
            if row_label == 'score mae':
                score_mae[opt].append(float(row[opt]))

for idx,ac in enumerate(achans):
    ldir = log_dir.format(suffixes[idx])
    with open(os.path.join(ldir,'aa_emb_strun_summ.csv'),'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for opt in opts_labels.keys():
                row_label = row['ML method']
                if row_label == 'score r2':
                    score_r2[opt].append(float(row[opt]))
                if row_label == 'score mae':
                    score_mae[opt].append(float(row[opt]))

text = ''.join(['& FS ']+['& AA{} '.format(ac) for ac in achans])
text += ''.join(['& FS ']+['& AA{} '.format(ac) for ac in achans]+['\\\\\\hline'])
print(text)
for opt in opts_labels.keys():
    val = max(score_r2[opt])
    text = ' {} '.format(opts_labels[opt])
    for r2 in score_r2[opt]:
        #if r2 == val:
        #    text += ' & \\textbf{{{:.3f}}} '.format(r2)
        #else:
        text += ' & {:.3f} '.format(r2)
    
    val = min(score_mae[opt])
    for mae in score_mae[opt]:
        #if mae == val:
        #    text += ' & \\textbf{{{:.2f}}} '.format(mae)
        #else:
        text += ' & {:.2f} '.format(mae)
    text += '\\\\\\hline'
    print(text)

