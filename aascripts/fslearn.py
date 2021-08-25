import numpy as np
import os
import csv
from autoatlas.rlearn import Predictor
from .cliargs import get_args
from .rlargs import RLEARN_ARGS

def get_subj_vals(task_tag,gt_filen,fture_filen):
    fture_tags = []
    with open(fture_filen,mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            assert len(row)==1
            fture_tags.append(row[0])

    #gt_file = 'hcpdata/unrestricted_kaplan7_4_1_2019_18_31_31.csv'
    gt_vals = {} #subject tag key and performance/category value 
    fture_vals = {} #subject tag key and performance/category value 
    with open(gt_filen,mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for i,row in enumerate(csv_reader):
            row = np.array(row)
            if i==0:
                gt_idx = [j for j in range(len(row)) if row[j]==task_tag]
                assert len(idx)==1
                gt_idx = idx[0]
                fs_idx = [j for j in range(len(row)) if row[j] in fture_tags]
                #print(labidx,row[labidx])
            else:
                if row[idx] != '' and np.all(row[fs_idx]!=''):
                    gt_vals.update({str(row[0]):row[idx]})
                    fture_vals.update({str(row[0]):row[fs_idx]})
    return gt_vals,fture_vals

def get_dataIO(savedir,only_embed,gtruths,task_type):
    files_code = [os.path.join(savedir,f) for f in os.listdir(savedir) if '_aacode.csv' in f]
    code_dict = read_code(files_code,only_embed)

    data_in,data_out,subjIDs = [],[],[]
    for key in code_dict.keys():
        if key in gtruths.keys():
            data_in.append(code_dict[key])
            data_out.append(gtruths[key]) 
            subjIDs.append(key)

    assert len(data_out)>300
    data_in = np.stack(data_in,axis=0)
    data_out = np.stack(data_out,axis=0)
    if task_type == 'regression':
        data_out = data_out.astype(float) 
    return data_in,data_out,subjIDs    
            
def write_csv(tag,mlm,subj,gtruth,pred,score,regsc,wdir,append=False):
    csv_data = {}
    if append == True:
        with open(os.path.join(wdir,'summ_rlearn_{}.csv'.format(tag)),'r',newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            for k in reader.fieldnames:
                csv_data[k] = []
            for row in reader:
                for k,val in row.items():
                    csv_data[k].append(val)

    csv_data['ML method'],csv_data[mlm] = [],[]
    for k in score.keys():
        csv_data['ML method'].append('Tot score {}'.format(k))
        csv_data[mlm].append('{:.6e}'.format(score[k]))

    for k in regsc.keys():
        for i in range(len(regsc[k])):
            csv_data['ML method'].append('Rg{} score {}'.format(i,k))       
            csv_data[mlm].append('{:.6e}'.format(regsc[k][i]))
 
    with open(os.path.join(wdir,'summ_rlearn_{}.csv'.format(tag)),'w',newline='') as csv_file:
        writer = csv.DictWriter(csv_file,fieldnames=csv_data.keys())
        writer.writeheader()
        for i in range(len(csv_data['ML method'])):
            writer.writerow({k:csv_data[k][i] for k in csv_data.keys()})

    fields = ['method','prediction','ground-truth']
    for idx,ID in enumerate(subj):
        csv_data = {}
        if append == True:
            with open(os.path.join(wdir,'{}_rlearn_{}.csv'.format(ID,tag)),'r',newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                for k in reader.fieldnames:
                    csv_data[k] = []
                for row in reader:
                    for k,val in row.items():
                        csv_data[k].append(val)

        csv_data['ML method'] = ['prediction','ground-truth']
        csv_data[mlm] = ['{:.6e}'.format(pred[idx]),'{:.6e}'.format(gtruth[idx])]
            
        with open(os.path.join(wdir,'{}_rlearn_{}.csv'.format(ID,tag)),'w',newline='') as csv_file:
            writer = csv.DictWriter(csv_file,fieldnames=csv_data.keys())
            writer.writeheader()
            for i in range(len(csv_data['ML method'])):
                writer.writerow({k:csv_data[k][i] for k in csv_data.keys()})
         
def main():
    extra_args = {
    'tag':[str,'Predict for chosen tag.'],
    'type':[str,'Must be either regression or classification'],
    'gtfile':[str,'File containing the ground-truth data.'],
    'fsfile':[str,'File containing list of features used for prediction.']}
    ARGS = get_args(extra_args)
  
    gtruths,features = get_subj_vals(ARGS['tag'],ARGS['gtfile'],ARGS['fsfile'])
    test_in,test_out,test_subj = get_dataIO(ARGS['test_savedir'],ARGS['only_embed'],gtruths,ARGS['type'])

    for i,rlarg in enumerate(RLEARN_ARGS):
        if ARGS['type'] == rlarg['type']:
            np.random.seed(0)

            ptor = Predictor(rlarg['estimator'])
            ptor.train(train_in,train_out)

            train_pred = ptor.predict(train_in)
            train_score,train_regsc = {},{}
            for key,met in rlarg['scorers'].items():
                train_score[key] = ptor.score(train_in,train_out,met)
            for key,rsc in rlarg['region_scorers'].items():
                train_regsc[key] = ptor.region_score(train_in,train_out,rlarg['scorers'][key],rsc,n_repeats=100)
            write_csv(ARGS['tag'],rlarg['tag'],train_subj,train_out,train_pred,train_score,train_regsc,ARGS['train_savedir'],i!=0)

            test_pred = ptor.predict(test_in)
            test_score,test_regsc = {},{}
            for key,met in rlarg['scorers'].items():
                test_score[key] = ptor.score(test_in,test_out,met)
            for key,rsc in rlarg['region_scorers'].items():
                test_regsc[key] = ptor.region_score(test_in,test_out,rlarg['scorers'][key],rsc,n_repeats=100)
            write_csv(ARGS['tag'],rlarg['tag'],test_subj,test_out,test_pred,test_score,test_regsc,ARGS['test_savedir'],i!=0)


