import numpy as np
import os
import csv
import argparse
from utils_pred import train_inf

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir',type=str,default='./checkpoints/',help='Directory for storing run time data')
parser.add_argument('--pred_file',type=str,default='pred_aa',help='Prefix for filename with predicted data')
parser.add_argument('--opt',type=str,default='lin',help='Optimizer chosen between dummy, lin, ransac, nneigh, boost, svm, lasso, or mlp')
parser.add_argument('--tag',type=str,default=None,help='Predict for chosen tag. If not specified, run for all.')
ARGS = parser.parse_args()

train_folder = os.path.join(ARGS.log_dir,'train_aa')
test_folder = os.path.join(ARGS.log_dir,'test_aa')
pred_folder = os.path.join(ARGS.log_dir,'pred_aa')
gt_file = 'hcpdata/unrestricted_kaplan7_4_1_2019_18_31_31.csv' 

if not os.path.exists(pred_folder):
    os.makedirs(pred_folder) 

if ARGS.tag is None:
    column_tags = ['Gender','Strength_Unadj','Endurance_Unadj','NEORAW_01']
    #column_tags = ['Gender','Age','Strength_Unadj','Strength_AgeAdj','PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','IWRD_TOT','ListSort_Unadj','ER40_CR','LifeSatisf_Unadj','Endurance_Unadj','Dexterity_Unadj','NEORAW_01','NEORAW_02','NEORAW_03','NEORAW_04','NEORAW_05','NEORAW_06','NEORAW_07','NEORAW_08','NEORAW_09','NEORAW_10']
else:
    column_tags = [ARGS.tag]

#with open('hcpdata/targets.csv','r') as csv_file:
#    csv_reader = csv.reader(csv_file)
#    for row in csv_reader:
#        assert len(row)==1
#        column_tags.append(row[0])  

inputs = np.load(os.path.join(train_folder,'train_inf_inps.npz'))
train_neigh_sims = inputs['neigh_sims']
train_seg_probs = inputs['seg_probs']
train_emb_codes = inputs['emb_codes']
train_ids = inputs['ids']
train_num = len(train_ids)

inputs = np.load(os.path.join(test_folder,'test_inf_inps.npz'))
test_neigh_sims = inputs['neigh_sims']
test_seg_probs = inputs['seg_probs']
test_emb_codes = inputs['emb_codes']
test_ids = inputs['ids']
test_num = len(test_ids)

num_labels = train_seg_probs.shape[1]
print('Number of labels is {}'.format(num_labels))

row_tags = ['all_seg_emb','all_seg','all_emb']
for i in range(num_labels):
    row_tags += ['lab{}_seg_emb'.format(i),'lab{}_seg'.format(i),'lab{}_emb'.format(i)] 

for cid,tag in enumerate(column_tags):
    print('Evaluating prediction performance for {}'.format(tag))
    gtruths = {} 
    with open(gt_file,mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for i,row in enumerate(csv_reader):
            row = np.array(row)
            if i==0:
                idx = [j for j in range(len(row)) if row[j]==tag]
                assert len(idx)==1
                idx = idx[0]
                #print(labidx,row[labidx])
            else:
                if row[idx] != '':
                    gtruths.update({str(row[0]):row[idx]})

    train_out,train_mask = [],[]
    for i in range(train_ids.size):
        if train_ids[i] in gtruths.keys(): 
            train_out.append(gtruths[train_ids[i]])
            train_mask.append(True)
        else:
            train_mask.append(False)

    test_out,test_mask = [],[]
    for i in range(test_ids.size):
        if test_ids[i] in gtruths.keys(): 
            test_out.append(gtruths[test_ids[i]])
            test_mask.append(True)
        else:
            test_mask.append(False)

    assert len(train_out)>300
    assert len(test_out)>300
#    print('Number of train data is {}'.format(len(train_out)))
#    print('Number of test data is {}'.format(len(test_out)))

    train_out = np.array(train_out)
    test_out = np.array(test_out)
    train_mask = np.array(train_mask)
    test_mask = np.array(test_mask)
    if tag != 'Gender' and tag != 'Age' and 'NEORAW_' not in tag:
        train_out = train_out.astype(float)    
        test_out = test_out.astype(float)    
        train_pred = np.zeros((len(row_tags),len(train_out)),dtype=np.float32)
        test_pred = np.zeros((len(row_tags),len(test_out)),dtype=np.float32)
        inf_type = 'regressor'
    else:
        train_pred = np.zeros((len(row_tags),len(train_out)),dtype=str)
        test_pred = np.zeros((len(row_tags),len(test_out)),dtype=str)
        inf_type = 'classifier'

    rid = 0
#    print(rid)
    train_in = np.concatenate((np.reshape(train_neigh_sims,(train_num,-1)),np.reshape(train_seg_probs,(train_num,-1)),np.reshape(train_emb_codes,(train_num,-1))),axis=-1)
    test_in = np.concatenate((np.reshape(test_neigh_sims,(test_num,-1)),np.reshape(test_seg_probs,(test_num,-1)),np.reshape(test_emb_codes,(test_num,-1))),axis=-1)
    train_pred[rid],test_pred[rid],_ = train_inf(inf_type,ARGS.opt,train_in,train_out,test_in,train_mask=train_mask,test_mask=test_mask,test_output=test_out)
    #print(train_in.shape,test_in.shape)

    rid = rid+1
#    print(rid)
    train_in = np.concatenate((np.reshape(train_neigh_sims,(train_num,-1)),np.reshape(train_seg_probs,(train_num,-1))),axis=-1)
    test_in = np.concatenate((np.reshape(test_neigh_sims,(test_num,-1)),np.reshape(test_seg_probs,(test_num,-1))),axis=-1)
    train_pred[rid],test_pred[rid],_ = train_inf(inf_type,ARGS.opt,train_in,train_out,test_in,train_mask=train_mask,test_mask=test_mask,test_output=test_out)
    #print(train_in.shape,test_in.shape)

    rid = rid+1
#    print(rid)
    train_in = np.reshape(train_emb_codes,(train_num,-1))
    test_in = np.reshape(test_emb_codes,(test_num,-1))
    train_pred[rid],test_pred[rid],_ = train_inf(inf_type,ARGS.opt,train_in,train_out,test_in,train_mask=train_mask,test_mask=test_mask,test_output=test_out)
    #print(train_in.shape,test_in.shape)

    for i in range(num_labels):
        rid = rid+1
#        print(rid)
        train_in = np.concatenate((np.reshape(train_neigh_sims[:,i],(train_num,-1)),np.reshape(train_seg_probs[:,i],(train_num,-1)),np.reshape(train_emb_codes[:,i],(train_num,-1))),axis=-1)
        test_in = np.concatenate((np.reshape(test_neigh_sims[:,i],(test_num,-1)),np.reshape(test_seg_probs[:,i],(test_num,-1)),np.reshape(test_emb_codes[:,i],(test_num,-1))),axis=-1)
        train_pred[rid],test_pred[rid],_ = train_inf(inf_type,ARGS.opt,train_in,train_out,test_in,train_mask=train_mask,test_mask=test_mask,test_output=test_out)
        #print(train_in.shape,test_in.shape)

        rid = rid+1
#        print(rid)
        train_in = np.concatenate((np.reshape(train_neigh_sims[:,i],(train_num,-1)),np.reshape(train_seg_probs[:,i],(train_num,-1))),axis=-1)
        test_in = np.concatenate((np.reshape(test_neigh_sims[:,i],(test_num,-1)),np.reshape(test_seg_probs[:,i],(test_num,-1))),axis=-1)
        train_pred[rid],test_pred[rid],_ = train_inf(inf_type,ARGS.opt,train_in,train_out,test_in,train_mask=train_mask,test_mask=test_mask,test_output=test_out)
        #print(train_in.shape,test_in.shape)

        rid = rid+1
#        print(rid)
        train_in = np.reshape(train_emb_codes[:,i],(train_num,-1))
        test_in = np.reshape(test_emb_codes[:,i],(test_num,-1))
        train_pred[rid],test_pred[rid],_ = train_inf(inf_type,ARGS.opt,train_in,train_out,test_in,train_mask=train_mask,test_mask=test_mask,test_output=test_out)
        #print(train_in.shape,test_in.shape)
        #mcls = 'M' if np.sum(train_outputs[:,0]=='M')>np.sum(train_outputs[:,0]=='F') else 'F'
        #print('Test majority',np.mean(test_outputs[:,0]==mcls))

    np.savez(os.path.join(pred_folder,'{}_tag{}_opt{}.npz'.format(ARGS.pred_file,tag,ARGS.opt)),train_pred=train_pred,train_true=train_out,test_pred=test_pred,test_true=test_out,row_tags=row_tags,column_tags=tag,train_ids=train_ids,test_ids=test_ids)
