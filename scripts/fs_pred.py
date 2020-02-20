import numpy as np
import os
import csv
from utils_pred import train_inf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir',type=str,default='./checkpoints/',help='Directory for storing run time data')
parser.add_argument('--pred_file',type=str,default='pred_fs',help='Prefix for filename with predicted data')
parser.add_argument('--opt',type=str,default='lin',help='Optimizer chosen between dummy, lin, ransac, nneigh, boost, svm, or mlp')
parser.add_argument('--tag',type=str,default=None,help='Predict for chosen tag. If not specified, run for all.')
ARGS = parser.parse_args()

pred_folder = os.path.join(ARGS.log_dir,'pred_aa')
feature_file = 'hcpdata/fs_features.csv'
gt_file = 'hcpdata/unrestricted_kaplan7_4_1_2019_18_31_31.csv' 
train_folder = os.path.join(ARGS.log_dir,'train_aa')
test_folder = os.path.join(ARGS.log_dir,'test_aa')
if not os.path.exists(pred_folder):
    os.makedirs(pred_folder) 

if ARGS.tag is None:
    column_tags = ['Gender','Strength_Unadj','Endurance_Unadj','NEORAW_01']
    #column_tags = ['Gender','Age','Strength_Unadj','Strength_AgeAdj','PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','IWRD_TOT','ListSort_Unadj','ER40_CR','LifeSatisf_Unadj','Endurance_Unadj','Dexterity_Unadj','NEORAW_01','NEORAW_02','NEORAW_03','NEORAW_04','NEORAW_05','NEORAW_06','NEORAW_07','NEORAW_08','NEORAW_09','NEORAW_10']
else:
    column_tags = [ARGS.tag]

inputs = np.load(os.path.join(train_folder,'train_inf_inps.npz'))
train_ids = inputs['ids']
inputs = np.load(os.path.join(test_folder,'test_inf_inps.npz'))
test_ids = inputs['ids']

feature_tags = []
with open(feature_file,'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        assert len(row)==1
        feature_tags.append(row[0])  

for cid,pred_tag in enumerate(column_tags):
    print('Evaluating prediction performance for {}'.format(pred_tag))
    features,gtruths = {},{} 
    with open(gt_file,mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for i,row in enumerate(csv_reader):
            #print(type(row))
            row = np.array(row)
            if i==0:
                gtidx = [j for j in range(len(row)) if row[j]==pred_tag]
                assert len(gtidx)==1
                gtidx = gtidx[0]
                feidx = np.array([j for j in range(len(row)) if row[j] in feature_tags])
                #print(labidx,row[labidx])
            else:
                if row[gtidx]!='' and np.all(row[feidx]!=''):
                    gtruths.update({str(row[0]):row[gtidx]})
                    features.update({str(row[0]):row[feidx]})

    train_out,train_in = [],[]
    for i in range(train_ids.size):
        if train_ids[i] in gtruths.keys(): 
            train_out.append(gtruths[train_ids[i]])
            train_in.append(features[train_ids[i]])

    test_out,test_in = [],[]
    for i in range(test_ids.size):
        if test_ids[i] in gtruths.keys(): 
            test_out.append(gtruths[test_ids[i]])
            test_in.append(features[test_ids[i]])

    assert len(train_out)>300
    assert len(test_out)>300

#    print('Number of train data is {}'.format(len(train_out)))
#    print('Number of test data is {}'.format(len(test_out)))

    train_out = np.array(train_out)
    test_out = np.array(test_out)
    train_in = np.stack(train_in,axis=0).astype(float)
    test_in = np.stack(test_in,axis=0).astype(float)
    if pred_tag != 'Gender' and pred_tag != 'Age' and 'NEORAW_' not in pred_tag:
        train_out = train_out.astype(float)    
        test_out = test_out.astype(float)    
        train_pred = np.zeros(len(train_out),dtype=np.float32)
        test_pred = np.zeros(len(test_out),dtype=np.float32)
        inf_type = 'regressor'
    else:
        train_pred = np.zeros(len(train_out),dtype=str)
        test_pred = np.zeros(len(test_out),dtype=str)
        inf_type = 'classifier'

    train_mask = np.arange(0,train_in.shape[0],1,dtype=int)
    test_mask = np.arange(0,test_in.shape[0],1,dtype=int)
    train_pred,test_pred,_ = train_inf(inf_type,ARGS.opt,train_in,train_out,test_in,train_mask=train_mask,test_mask=test_mask,test_output=test_out)
    #print(train_in.shape,test_in.shape)

    np.savez(os.path.join(pred_folder,'{}_tag{}_opt{}.npz'.format(ARGS.pred_file,pred_tag,ARGS.opt)),train_pred=train_pred[np.newaxis],train_true=train_out,test_pred=test_pred[np.newaxis],test_true=test_out,column_tags=pred_tag,row_tags='all',train_ids=train_ids,test_ids=test_ids)

