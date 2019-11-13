import numpy as np
import os

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05/pred_aa'
aafile = 'perfm_aa{}_tag{}_opt{}.npz'
fsfile = 'perfm_fs_tag{}_opt{}.npz'

opts = ['lin']
opts_labels = ['Linear','NNeighbor','Boosting','MLPerceptron']
achans = [4,8,16]
column_tag = 'Strength_Unadj'
num_labels = 16

aamet0 = np.zeros((len(achans),len(opts),num_labels)) 
aamet1 = np.zeros((len(achans),len(opts),num_labels)) 
for i,ac in enumerate(achans):
    for j,op in enumerate(opts):
        filen = os.path.join(log_dir.format(ac),aafile.format(ac,column_tag,op))
        #print('Reading {}'.format(filen))
        perf = np.load(filen)
        aamet0[i,j] = perf['test_perfs'][5::3,0]            
        aamet1[i,j] = perf['test_perfs'][5::3,1]            

strg = ' &  '
for ac in achans:
    strg += '& AA{} '.format(ac)
strg += '\\\\\hline'
print(strg)

for lid in range(num_labels):
    for j,op in enumerate(opts):
        strg = 'Label {} & {} '.format(lid,opts_labels[j])
        for k,ac in enumerate(achans):
            strg += '& {:.3f} / {:.3f} '.format(aamet0[k,j,lid],aamet1[k,j,lid])    
        strg += '\\\\\hline'
        print(strg) 
                    
