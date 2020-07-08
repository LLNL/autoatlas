from .cliargs import get_args,write_args
from .utils import get_dataset
import numpy as np
import csv
import os
from autoatlas.dirnn import DirPredNN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def get_dout(out_file,smpl_list,target,task_type):
    samples = []
    with open(smpl_list,'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            assert len(row)==1
            samples.append(row[0])
    
    data_out = []
    for smpl in samples:
        with open(out_file.format(smpl),'r') as csv_file:
            reader = csv.reader(csv_file)
            for i,row in enumerate(reader):
                assert len(row)==2
                if i!=0 and row[0]==target:
                    data_out.append(row[1]) 

    assert len(data_out)==len(samples)
    data_out = np.stack(data_out,axis=0)
    if task_type == 'regression':
        data_out = data_out.astype(float)
    return data_out,samples 

def process_data(dirNN,smpl_subj,dvol_filen,dmask_filen,gt_all,pred_filen,summ_filen,task,labels,scaler):
    batch = dirNN.ARGS['batch']
    pred_all = []
    for i in range(0,len(smpl_subj),batch):
        smpl_slice = smpl_subj[i:i+batch]
        gt_slice = gt_all[i:i+batch]

        smpl_dataset,data_fin,mask_fin = get_dataset(samples=smpl_slice,data_filename=dvol_filen,mask_filename=dmask_filen,targets=gt_slice,task=task,labels=labels)
        pred_out,data_fout,mask_fout = dirNN.process(smpl_dataset,ret_input=False)
        assert data_fin==data_fout and len(data_fin)<=batch
        assert mask_fin==mask_fout and len(mask_fin)<=batch

        pred_all.extend(pred_out)
        for idx,(ID,gt) in enumerate(zip(smpl_slice,gt_slice)):
            print(ID)
            filen = pred_filen.format(ID)
            folder = os.path.split(filen)
            assert len(folder)==2
            folder = folder[0]
            os.makedirs(folder,exist_ok=True)
            
            with open(filen,'w',newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['pred',pred_out[idx]])
                writer.writerow(['gt',gt])

    gt_all = np.array(gt_all)
    pred_all = np.array(pred_all)
    assert gt_all.ndim==1
    assert pred_all.ndim==1
    pred_all = scaler.inverse_transform(pred_all.reshape(-1,1)).squeeze()
    if task == 'regression':
        r2 = metrics.r2_score(gt_all,pred_all)    
        mae = metrics.mean_absolute_error(gt_all,pred_all)
        mape = 100*np.mean(np.absolute(pred_all-gt_all)/gt_all) 
        with open(summ_filen,'w',newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['r2',r2])
            writer.writerow(['mae',mae])
            writer.writerow(['mape',mape])
    else:
        bacc = metrics.balanced_accuracy_score(gt_all,pred_all)
        acc = metrics.accuracy_score(gt_all,pred_all)
        with open(summ_filen,'w',newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['bacc',bacc])
            writer.writerow(['acc',acc])

def main():
    ARGS_dict = {'ckpt':[str,'File for storing run time data.'],
                'target':[str,'Name of parameter to be predicted.'],
                'task':[str,'Choose between regression and classification.'],
                'train_in':[str,'Filepath to input volume from training dataset.'],
                'train_out':[str,'Filename of output targets for training.'],
                'train_mask':[str,'Filepath of mask for training dataset.'],
                'train_list':[str,'File containing list of training samples.'],
                'train_pred':[str,'File to store predicted values from train.'],
                'train_summ':[str,'File to store ML performance metrics from train.'],
                'test_in':[str,'Filepath to input volume from testing dataset.'],
                'test_out':[str,'Filename of output targets for testing.'],
                'test_mask':[str,'Filepath of mask for testing dataset.'],
                'test_list':[str,'File containing list of testing samples.'],
                'test_pred':[str,'File to store predicted values from test.'],
                'test_summ':[str,'File to store ML performance metrics from test.'],
                'load_epoch':[int,'Model epoch to load. If negative, does not load model.'],
                'device':[str,'Choose between cuda or cpu.']}
    
    ARGS = get_args(ARGS_dict)
    cli_file = os.path.split(ARGS['cli_args'])[-1]
    write_args(ARGS,ARGS['cli_save'])
    
    if ARGS['load_epoch'] >= 0:
        dirNN = DirPredNN(device=ARGS['device'],load_ckpt_epoch=ARGS['load_epoch'],ckpt_file=ARGS['ckpt'])
    else:
        raise ValueError('load_epoch must be specified')
    
    if ARGS['train_list'] is not None:
        train_out,train_subj = get_dout(ARGS['train_out'],ARGS['train_list'],ARGS['target'],ARGS['task'])
        labels = np.unique(train_out)
        scaler = StandardScaler(with_mean=True,with_std=True)
        scaler.fit(train_out.reshape(-1,1))
        #train_out = scaler.transform(train_out.reshape(-1,1)).squeeze()
        process_data(dirNN,train_subj,ARGS['train_in'],ARGS['train_mask'],train_out,ARGS['train_pred'],ARGS['train_summ'],ARGS['task'],labels,scaler)

    if ARGS['test_list'] is not None:
        test_out,test_subj = get_dout(ARGS['test_out'],ARGS['test_list'],ARGS['target'],ARGS['task'])
        #test_out = scaler.transform(test_out.reshape(-1,1)).squeeze()
        process_data(dirNN,test_subj,ARGS['test_in'],ARGS['test_mask'],test_out,ARGS['test_pred'],ARGS['test_summ'],ARGS['task'],labels,scaler)

if __name__ == "__main__": 
    main() 
