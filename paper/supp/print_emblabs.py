import numpy as np
import os

log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc{}_11_labels16_smooth0.1_devr1.0_freqs0.05/pred_aa'
aafile = 'perfm_aa{}_tag{}_opt{}.npz'
fsfile = 'perfm_fs_tag{}_opt{}.npz'

opts = ['lin','nneigh','svm','mlp']
opts_labels = ['Lin','NNbor','SVM','MLP']
achans = [4,8,16]
column_tag = 'Strength_Unadj'
num_labels = 16

#aamet0 = np.zeros((len(achans),len(opts),num_labels)) 
aamet1 = np.zeros((len(achans),len(opts),num_labels)) 
aapre1 = np.empty((len(achans),len(opts),num_labels),dtype=object)
aasuf1 = np.empty((len(achans),len(opts),num_labels),dtype=object)
for i,ac in enumerate(achans):
    for j,op in enumerate(opts):
        filen = os.path.join(log_dir.format(ac),aafile.format(ac,column_tag,op))
        #print('Reading {}'.format(filen))
        perf = np.load(filen)
#        aamet0[i,j] = perf['test_perfs'][3::3,0]            
        aamet1[i,j] = perf['test_perfs'][5::3,1]           
        for k in range(num_labels):
            if aamet1[i,j,k] > 0.2:
                aapre1[i,j,k] = '\\textbf{'
                aasuf1[i,j,k] = '}'
            else:
                aapre1[i,j,k] = ''
                aasuf1[i,j,k] = ''

labstrg = ''
for i in range(num_labels):
    labstrg += '& $e_{'+str(i+1)+'}$ '
labstrg += '\\\\\hline'
for i,ac in enumerate(achans):
    print('\hline \multicolumn{17}{|c|}{Number of autoencoder channels, $C_a = '+str(ac)+'$}\\\\\hline')
    print(labstrg)
    for j,op in enumerate(opts):
        strg = '{} '.format(opts_labels[j])
        for k in range(num_labels):
            strg += '& {}{:.2f}{} '.format(aapre1[i,j,k],aamet1[i,j,k],aasuf1[i,j,k])    
        strg += '\\\\\hline'
        print(strg) 
                    
