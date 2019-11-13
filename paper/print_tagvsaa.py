import numpy as np
import os

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05/pred_aa'
aafile = 'perfm_aa{}_tag{}_opt{}.npz'
fsfile = 'perfm_fs_tag{}_opt{}.npz'

opts = ['lin','nneigh','boost','mlp']
opts_labels = ['Linear','NNeighbor','Boosting','MLPerceptron']
achans = [4,8,16]
column_tags = ['Strength_Unadj','Endurance_Unadj','Gender','NEORAW_01','NEORAW_02']
#column_tags = ['Strength_Unadj','Endurance_Unadj','Gender','NEORAW_01','NEORAW_02','NEORAW_03','NEORAW_04','NEORAW_05','NEORAW_06','NEORAW_07','NEORAW_08','NEORAW_09','NEORAW_10']

aamet0 = np.zeros((len(column_tags),len(opts),len(achans))) 
aamet1 = np.zeros((len(column_tags),len(opts),len(achans))) 
for i,tag in enumerate(column_tags):
    for j,op in enumerate(opts):
        for k,ac in enumerate(achans):
            filen = os.path.join(log_dir.format(ac),aafile.format(ac,tag,op))
            #print('Reading {}'.format(filen))
            perf = np.load(filen)
            aamet0[i,j,k] = perf['test_perfs'][2,0]            
            aamet1[i,j,k] = perf['test_perfs'][2,1]            

fsmet0 = np.zeros((len(column_tags),len(opts)))
fsmet1 = np.zeros((len(column_tags),len(opts)))
for i,tag in enumerate(column_tags):
    for j,op in enumerate(opts):
        filen = os.path.join(log_dir.format(4),fsfile.format(tag,op))
        #print('Reading {}'.format(filen))
        perf = np.load(filen)
        fsmet0[i,j] = perf['test_perfs'][0,0]
        fsmet1[i,j] = perf['test_perfs'][0,1]

strg = ' &  & FS '
for ac in achans:
    strg += '& AA{} '.format(ac)
strg += '& FS '
for ac in achans:
    strg += '& AA{} '.format(ac)
strg += '\\\\\hline'
print(strg)

for i,tag in enumerate(column_tags):
    for j,op in enumerate(opts):
        if '_' in tag:
            tag = tag.split('_')
            tag = tag[0]+'-'+tag[1]
        strg = '{} & {} & {:.3f} '.format(tag,opts_labels[j],fsmet0[i,j])
        for k,ac in enumerate(achans):
            strg += '& {:.3f} '.format(aamet0[i,j,k])    
        strg += '& {:.3f} '.format(fsmet1[i,j])
        for k,ac in enumerate(achans):
            strg += '& {:.3f} '.format(aamet1[i,j,k])    
        strg += '\\\\\hline'
        print(strg) 
                
