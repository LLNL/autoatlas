import os
import h5py
import numpy as np
from sklearn.metrics import r2_score,accuracy_score,balanced_accuracy_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir',type=str,default='./checkpoints/',help='Directory for storing run time data')
ARGS = parser.parse_args()

log_dir = ARGS.log_dir
#log_dir = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05'

pred_folder = os.path.join(log_dir,'pred_aa')
pred_files = [f for f in os.listdir(pred_folder) if f[-4:]=='.npz' and f[:5]=='pred_']

for filen in pred_files:
    npz_file = np.load(os.path.join(pred_folder,filen))
    tag = str(npz_file['column_tags'])

    train_pred = npz_file['train_pred']  
    train_true = npz_file['train_true']
    if tag!='Gender' and tag!='Age' and 'NEORAW_' not in tag:
        norm = np.mean(train_true)
    train_perfs = np.zeros((train_pred.shape[0],2),dtype=float)
    for i in range(train_pred.shape[0]):
        perf = np.zeros(2,dtype=float)
        if tag!='Gender' and tag!='Age' and 'NEORAW_' not in tag:
            perf[0] = np.sqrt(np.mean((train_true-train_pred[i])**2))/norm
            perf[1] = r2_score(train_true,train_pred[i])
        else:
            perf[0] = accuracy_score(train_true,train_pred[i])
            perf[1] = balanced_accuracy_score(train_true,train_pred[i])
        train_perfs[i] = perf 
    
    test_pred = npz_file['test_pred']  
    test_true = npz_file['test_true'] 
    test_perfs = np.zeros((test_pred.shape[0],2),dtype=float)
    for i in range(test_pred.shape[0]):
        perf = np.zeros(2,dtype=float)
        if tag!='Gender' and tag!='Age' and 'NEORAW_' not in tag:
            perf[0] = np.sqrt(np.mean((test_true-test_pred[i])**2))/norm
            perf[1] = r2_score(test_true,test_pred[i])
        else:
            perf[0] = accuracy_score(test_true,test_pred[i])
            perf[1] = balanced_accuracy_score(test_true,test_pred[i])
        test_perfs[i] = perf 
   
    wrfilen = os.path.join(pred_folder,'perfm'+filen[4:]) 
    print('Writing file {}'.format(wrfilen))
    np.savez(wrfilen,train_perfs=train_perfs,test_perfs=test_perfs,row_tags=npz_file['row_tags'],column_tags=tag) 
