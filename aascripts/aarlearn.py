import numpy as np
import os
import csv
from autoatlas.rlearn import Predictor
from aascripts.cliargs import get_parser,get_args
from aascripts.rlargs import RLEARN_ARGS

def aarlearn_parser(ret_dict=False):
    extra_args = {'target':[str,'Name of parameter to be predicted.'],
    'task':[str,'Choose between regression or classification.'],
    'train_in':[str,'Filename of input features for training.'],
    'train_out':[str,'Filename of output targets for training.'],
    'train_list':[str,'File containing list of training samples.'],
    'train_pred':[str,'File to store predicted values from train.'],
    'train_summ':[str,'File to store ML performance metrics from train.'],
    'test_in':[str,'Filename of input features for testing.'],
    'test_out':[str,'Filename of output targets for testing.'],
    'test_list':[str,'File containing list of testing samples.'],
    'test_pred':[str,'File to store predicted values from test.'],
    'test_summ':[str,'File to store ML performance metrics from test.'],
    'no_frank':[bool,'If True, do not compute feature ranks.']
    }
    return get_parser(extra_args, ret_dict)

def get_dataIO(in_file,out_file,smpl_list,target,task_type):
    samples = []
    with open(smpl_list,'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            assert len(row)==1
            samples.append(row[0])
    
    data_in,data_out = [],[]
    for smpl in samples:
        with open(in_file.format(smpl),'r') as csv_file:
            reader = csv.reader(csv_file)
            features = []
            for i,row in enumerate(reader):
                if i!=0:
                    features.append(row[1:])
        data_in.append(features) 
        
        with open(out_file.format(smpl),'r') as csv_file:
            reader = csv.reader(csv_file)
            for i,row in enumerate(reader):
                assert len(row)==2
                if i!=0 and row[0]==target:
                    data_out.append(row[1]) 

    assert len(data_in)==len(data_out),'len(data_in)={},len(data_out)={}'.format(len(data_in),len(data_out))
    data_in = np.stack(data_in,axis=0).astype(float)
    data_out = np.stack(data_out,axis=0)
    if task_type == 'regression':
        data_out = data_out.astype(float)
    return data_in,data_out,samples 
            
def write_csv(mlm,summ_file,pred_file,subj,gtruth,pred,score,regsc,append=False):
    csv_data = {}
    if append == True:
        with open(summ_file,'r',newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for k in reader.fieldnames:
                csv_data[k] = []
            for row in reader:
                for k,val in row.items():
                    csv_data[k].append(val)

    csv_data['ML method'],csv_data[mlm] = [],[]
    for k in score.keys():
        csv_data['ML method'].append('score {}'.format(k))
        csv_data[mlm].append('{:.6e}'.format(score[k]))

    if regsc is not None:
        for k in regsc.keys():
            for i in range(len(regsc[k])):
                csv_data['ML method'].append('fv{} imp {}'.format(i,k))       
                csv_data[mlm].append('{:.6e}'.format(regsc[k][i]))
 
    with open(summ_file,'w',newline='') as csv_file:
        writer = csv.DictWriter(csv_file,fieldnames=csv_data.keys())
        writer.writeheader()
        for i in range(len(csv_data['ML method'])):
            writer.writerow({k:csv_data[k][i] for k in csv_data.keys()})

    fields = ['method','pred','gtruth']
    for idx,ID in enumerate(subj):
        csv_data = {}
        if append == True:
            with open(pred_file.format(ID),'r',newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                for k in reader.fieldnames:
                    csv_data[k] = []
                for row in reader:
                    for k,val in row.items():
                        csv_data[k].append(val)

        csv_data['ML method'] = ['pred','gtruth']
        csv_data[mlm] = []
        if isinstance(pred[idx],str):
            csv_data[mlm].append('{}'.format(pred[idx]))
        else:
            csv_data[mlm].append('{:.6e}'.format(pred[idx]))
        
        if isinstance(pred[idx],str):
            csv_data[mlm].append('{}'.format(gtruth[idx]))
        else:
            csv_data[mlm].append('{:.6e}'.format(gtruth[idx]))
            
        with open(pred_file.format(ID),'w',newline='') as csv_file:
            writer = csv.DictWriter(csv_file,fieldnames=csv_data.keys())
            writer.writeheader()
            for i in range(len(csv_data['ML method'])):
                writer.writerow({k:csv_data[k][i] for k in csv_data.keys()})
         
def main():
    ARGS = get_args(*aarlearn_parser(ret_dict=True))
 
    train_in,train_out,train_subj = get_dataIO(ARGS['train_in'],ARGS['train_out'],ARGS['train_list'],ARGS['target'],ARGS['task'])
    test_in,test_out,test_subj = get_dataIO(ARGS['test_in'],ARGS['test_out'],ARGS['test_list'],ARGS['target'],ARGS['task'])

    pred_idx = 0
    for rlarg in RLEARN_ARGS:
        if ARGS['task'] == rlarg['task']:
            print(rlarg)
            np.random.seed(0)

            ptor = Predictor(rlarg['estimator'])
            ptor.train(train_in,train_out)

            train_pred = ptor.predict(train_in)
            train_score = {}
            for key,met in rlarg['scorers'].items():
                train_score[key] = ptor.score(train_in,train_out,met)
            if not ARGS['no_frank']:
                train_regsc = {}
                for key,rsc in rlarg['feature_scorers'].items():
                    train_regsc[key] = ptor.region_score(train_in,train_out,rlarg['scorers'][key],rsc,n_repeats=100)
            else:
                train_regsc = None
            write_csv(rlarg['tag'],ARGS['train_summ'],ARGS['train_pred'],train_subj,train_out,train_pred,train_score,train_regsc,pred_idx!=0)

            test_pred = ptor.predict(test_in)
            test_score = {}
            for key,met in rlarg['scorers'].items():
                test_score[key] = ptor.score(test_in,test_out,met)
            if not ARGS['no_frank']:
                test_regsc = {}
                for key,rsc in rlarg['feature_scorers'].items():
                    test_regsc[key] = ptor.region_score(test_in,test_out,rlarg['scorers'][key],rsc,n_repeats=100)
            else:
                test_regsc = None
            write_csv(rlarg['tag'],ARGS['test_summ'],ARGS['test_pred'],test_subj,test_out,test_pred,test_score,test_regsc,pred_idx!=0)
            pred_idx = pred_idx+1


