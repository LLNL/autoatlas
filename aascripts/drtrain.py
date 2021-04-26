from autoatlas.dirnn import DirPredNN
import csv
import numpy as np
import os
from .cliargs import get_args,write_args
from .utils import get_dataset
from sklearn.preprocessing import StandardScaler

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
            
def main():
    extra_args = {'target':[str,'Name of parameter to be predicted.'],
    'task':[str,'Choose between regression and classification.'],
    'train_in':[str,'Filename of input data for training.'],
    'train_out':[str,'Filename of output targets for training.'],
    'train_mask':[str,'Filename of masks for input volumes in training set.'],
    'train_list':[str,'File containing list of training samples.'],
    'test_in':[str,'Filename of input data for testing.'],
    'test_out':[str,'Filename of output targets for testing.'],
    'test_mask':[str,'Filename of masks for input volumes in test set.'],
    'test_list':[str,'File containing list of testing samples.'],
    'ckpt':[str,'File for storing run time data.'],
    'epochs':[int,'Number of epochs.'],
    'batch':[int,'Batch size.'],
    'lr':[float,'Learning rate.'],
    'wdcy':[float,'Weight decay.'],
    'load_epoch':[int,'Model epoch to load. If negative, does not load model.'],
    'in_chan':[int,'Number of channels in input image or volume.'],
    'in_dims':[str,'Dimensions separated by comma.'],
    'cnn_chan':[int,'Number of CNN channels.'],
    'cnn_depth':[int,'Depth of CNN.'],
    'device':[str,'Choose between cuda or cpu.'],
    }

    ARGS = get_args(extra_args)
    if not os.path.exists(ARGS['cli_args']):
        raise ValueError('{} not found'.format(ARGS['cli_args']))
    
    write_args(ARGS,ARGS['cli_save'])

    train_out,train_subj = get_dout(ARGS['train_out'],ARGS['train_list'],ARGS['target'],ARGS['task'])
    labels = np.unique(train_out)
    scaler = StandardScaler(with_mean=True,with_std=True)
    scaler.fit(train_out.reshape(-1,1))
    train_out = scaler.transform(train_out.reshape(-1,1)).squeeze()
    train_dataset,_,_ = get_dataset(train_subj,ARGS['train_in'],ARGS['train_mask'],targets=train_out,task=ARGS['task'],labels=labels)

    if ARGS['test_list'] is not None:
        test_out,test_subj = get_dout(ARGS['test_out'],ARGS['test_list'],ARGS['target'],ARGS['task'])
        test_out = scaler.transform(test_out.reshape(-1,1)).squeeze()
        test_dataset,_,_ = get_dataset(test_subj,ARGS['test_in'],ARGS['test_mask'],targets=test_out,task=ARGS['task'],labels=labels)

    #NN Model
    dims = [int(d) for d in ARGS['in_dims'].split(',')]
    if ARGS['load_epoch'] >= 0:
        dirNN = DirPredNN(sizes=dims,data_chan=ARGS['in_chan'],batch=ARGS['batch'],lr=ARGS['lr'],wdcy=ARGS['wdcy'],cnn_chan=ARGS['cnn_chan'],cnn_depth=ARGS['cnn_depth'],device=ARGS['device'],load_ckpt_epoch=ARGS['load_epoch'],ckpt_file=ARGS['ckpt'],task=ARGS['task'])
    elif ARGS['load_epoch'] == -1:
        dirNN = DirPredNN(sizes=dims,data_chan=ARGS['in_chan'],batch=ARGS['batch'],lr=ARGS['lr'],wdcy=ARGS['wdcy'],cnn_chan=ARGS['cnn_chan'],cnn_depth=ARGS['cnn_depth'],device=ARGS['device'],ckpt_file=ARGS['ckpt'],task=ARGS['task'])
    else:
        raise ValueError('load_epoch must be greater than or equal to -1')
    
    #Training
    prev_epoch = ARGS['load_epoch']
    for epoch in range(prev_epoch+1,ARGS['epochs']):
        print("Epoch {}".format(epoch))
        dirNN.train(train_dataset)
        if ARGS['test_list'] is not None:
            dirNN.test(test_dataset)
        dirNN.ckpt(epoch,ARGS['ckpt'])
        ARGS['load_epoch'] = epoch
        write_args(ARGS,ARGS['cli_save'])

