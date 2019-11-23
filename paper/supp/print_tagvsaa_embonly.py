import numpy as np
import os

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05/pred_aa'
aafile = 'perfm_aa{}_tag{}_opt{}.npz'
fsfile = 'perfm_fs_tag{}_opt{}.npz'

opts = ['lin','nneigh','svm','mlp']
opts_labels = ['Lin','NNbor','SVM','MLP']
achans = [4,8,16]
column_tags = ['Strength_Unadj','Endurance_Unadj','Gender','NEORAW_01']
#column_tags = ['Strength_Unadj','Endurance_Unadj','Gender','NEORAW_01','NEORAW_02','NEORAW_03','NEORAW_04','NEORAW_05','NEORAW_06','NEORAW_07','NEORAW_08','NEORAW_09','NEORAW_10']

aamet0 = np.zeros((len(column_tags),len(opts),len(achans)),dtype=float) 
aamet1 = np.zeros((len(column_tags),len(opts),len(achans)),dtype=float) 
for i,tag in enumerate(column_tags):
    for j,op in enumerate(opts):
        for k,ac in enumerate(achans):
            filen = os.path.join(log_dir.format(ac),aafile.format(ac,tag,op))
            #print('Reading {}'.format(filen))
            perf = np.load(filen)
            aamet0[i,j,k] = perf['test_perfs'][2,0]            
            aamet1[i,j,k] = perf['test_perfs'][2,1]            

fsmet0 = np.zeros((len(column_tags),len(opts)),dtype=float)
fsmet1 = np.zeros((len(column_tags),len(opts)),dtype=float)
for i,tag in enumerate(column_tags):
    for j,op in enumerate(opts):
        filen = os.path.join(log_dir.format(4),fsfile.format(tag,op))
        #print('Reading {}'.format(filen))
        perf = np.load(filen)
        fsmet0[i,j] = perf['test_perfs'][0,0]
        fsmet1[i,j] = perf['test_perfs'][0,1]

aapre0 = np.empty((len(column_tags),len(opts),len(achans)),dtype=object)
aasuf0 = np.empty((len(column_tags),len(opts),len(achans)),dtype=object)
aapre1 = np.empty((len(column_tags),len(opts),len(achans)),dtype=object)
aasuf1 = np.empty((len(column_tags),len(opts),len(achans)),dtype=object)
fspre0 = np.empty((len(column_tags),len(opts)),dtype=object)
fssuf0 = np.empty((len(column_tags),len(opts)),dtype=object)
fspre1 = np.empty((len(column_tags),len(opts)),dtype=object)
fssuf1 = np.empty((len(column_tags),len(opts)),dtype=object)
for i,tag in enumerate(column_tags):
    for j,op in enumerate(opts):
        if 'Unadj' in tag:
            val = np.min(np.concatenate((aamet0[i,j],fsmet0[i,j][np.newaxis]))) 
        else:
            val = np.max(np.concatenate((aamet0[i,j],fsmet0[i,j][np.newaxis]))) 
        for k,ac in enumerate(achans):
            aapre0[i,j,k] = "\\textbf{" if aamet0[i,j,k]==val else '' 
            aasuf0[i,j,k] = '}' if aamet0[i,j,k]==val else '' 
        fspre0[i,j] = "\\textbf{" if fsmet0[i,j]==val else '' 
        fssuf0[i,j] = '}' if fsmet0[i,j]==val else '' 
        
        val = np.max(np.concatenate((aamet1[i,j],fsmet1[i,j][np.newaxis]))) 
        for k,ac in enumerate(achans):
            aapre1[i,j,k] = "\\textbf{" if aamet1[i,j,k]==val else '' 
            aasuf1[i,j,k] = '}' if aamet1[i,j,k]==val else ''
        fspre1[i,j] = "\\textbf{" if fsmet1[i,j]==val else '' 
        fssuf1[i,j] = '}' if fsmet1[i,j]==val else '' 

strg = ' &  & FS '
for ac in achans:
    strg += '& AA, $C_a{}$ '.format(ac)
strg += '& FS '
for ac in achans:
    strg += '& AA, $C_a{}$ '.format(ac)
strg += '\\\\\hline'
print(strg)

for i,tag in enumerate(column_tags):
    for j,op in enumerate(opts):
        if '_' in tag:
            tag = tag.split('_')
            tag = tag[0]+'-'+tag[1]
        strg = '{} & {} & {}{:.4f}{} '.format(tag,opts_labels[j],fspre0[i,j],fsmet0[i,j],fssuf0[i,j])
        for k,ac in enumerate(achans):
            strg += '& {}{:.4f}{} '.format(aapre0[i,j,k],aamet0[i,j,k],aasuf0[i,j,k])    
        strg += '& {}{:.4f}{} '.format(fspre1[i,j],fsmet1[i,j],fssuf1[i,j])
        for k,ac in enumerate(achans):
            strg += '& {}{:.4f}{} '.format(aapre1[i,j,k],aamet1[i,j,k],aasuf1[i,j,k])    
        strg += '\\\\\hline'
        print(strg) 
                
