import numpy as np
import os
import csv
from autoatlas import Predictor
from .cliargs import get_args
from .rlargs import RLEARN_ARGS
from .cliargs import HELP_MSG_DICT as HELP

def read_code(filenames,only_embed):
    dict_codes = {}
    for filen in filenames:
        with open(filen,mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file,delimiter=',')
            num_embed = len([k for k in csv_reader.fieldnames if 'encoding ' in k])
            exp_fields = ['region','normalized volume','normalized surface area'] + ['encoding {}'.format(i) for i in range(num_embed)]
            assert csv_reader.fieldnames == exp_fields
            codes = []
            if only_embed:
                for row in csv_reader:
                    codes.append([row['embed_{}'.format(k)] for k in range(num_embed)])
            else:
                for row in csv_reader:
                    codes.append([row[key] for key in csv_reader.fieldnames if key!='region'])
        subj = os.path.split(filen)[-1].split('_')[0]
        dict_codes[subj] = np.stack(codes,axis=0)
    return dict_codes

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
    data_in = np.stack(data_in,axis=0)
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
        csv_data[mlm] = ['{:.6e}'.format(pred[idx]),'{:.6e}'.format(gtruth[idx])]
            
        with open(pred_file.format(ID),'w',newline='') as csv_file:
            writer = csv.DictWriter(csv_file,fieldnames=csv_data.keys())
            writer.writeheader()
            for i in range(len(csv_data['ML method'])):
                writer.writerow({k:csv_data[k][i] for k in csv_data.keys()})
         
def main():
    extra_args = {'target':[str,HELP['target_rlearn']],
    'task':[str,HELP['task_rlearn']],
    'train_in':[str,HELP['train_in_rlearn']],
    'train_out':[str,HELP['train_out_rlearn']],
    'train_list':[str,HELP['train_list']],
    'train_pred':[str,HELP['train_pred_rlearn']],
    'train_summ':[str,HELP['train_summ_rlearn']],
    'test_in':[str,HELP['test_in_rlearn']],
    'test_out':[str,HELP['test_out_rlearn']],
    'test_list':[str,HELP['test_list']],
    'test_pred':[str,HELP['test_pred_rlearn']],
    'test_summ':[str,HELP['test_summ_rlearn']]}
    ARGS = get_args(extra_args)
   
    train_in,train_out,train_subj = get_dataIO(ARGS['train_in'],ARGS['train_out'],ARGS['train_list'],ARGS['target'],ARGS['task'])
    test_in,test_out,test_subj = get_dataIO(ARGS['test_in'],ARGS['test_out'],ARGS['test_list'],ARGS['target'],ARGS['task'])

    for i,rlarg in enumerate(RLEARN_ARGS):
        if ARGS['task'] == rlarg['task']:
            np.random.seed(0)

            ptor = Predictor(rlarg['estimator'])
            ptor.train(train_in,train_out)

            train_pred = ptor.predict(train_in)
            train_score,train_regsc = {},{}
            for key,met in rlarg['scorers'].items():
                train_score[key] = ptor.score(train_in,train_out,met)
            for key,rsc in rlarg['feature_scorers'].items():
                train_regsc[key] = ptor.region_score(train_in,train_out,rlarg['scorers'][key],rsc,n_repeats=100)
            write_csv(rlarg['tag'],ARGS['train_summ'],ARGS['train_pred'],train_subj,train_out,train_pred,train_score,train_regsc,i!=0)

            test_pred = ptor.predict(test_in)
            test_score,test_regsc = {},{}
            for key,met in rlarg['scorers'].items():
                test_score[key] = ptor.score(test_in,test_out,met)
            for key,rsc in rlarg['feature_scorers'].items():
                test_regsc[key] = ptor.region_score(test_in,test_out,rlarg['scorers'][key],rsc,n_repeats=100)
            write_csv(rlarg['tag'],ARGS['test_summ'],ARGS['test_pred'],test_subj,test_out,test_pred,test_score,test_regsc,i!=0)


