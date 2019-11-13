import numpy as np
import os
import csv
from sklearn.linear_model import LogisticRegression,LinearRegression,RANSACRegressor,Lasso
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.svm import SVC,SVR
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.dummy import DummyClassifier,DummyRegressor
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
    column_tags = ['Gender','Age','Strength_Unadj','Strength_AgeAdj','PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','IWRD_TOT','ListSort_Unadj','ER40_CR','LifeSatisf_Unadj','Endurance_Unadj','Dexterity_Unadj','NEORAW_01','NEORAW_02','NEORAW_03','NEORAW_04','NEORAW_05','NEORAW_06','NEORAW_07','NEORAW_08','NEORAW_09','NEORAW_10']
else:
    column_tags = [ARGS.tag]

#column_tags = ['Gender','Strength_Unadj','Strength_AgeAdj','NEORAW_01','NEORAW_02','NEORAW_03','NEORAW_04','NEORAW_05','NEORAW_06','NEORAW_07','NEORAW_08','NEORAW_09','NEORAW_10','NEORAW_11','NEORAW_12','NEORAW_13','NEORAW_14','NEORAW_15','NEORAW_16','NEORAW_17','NEORAW_18','NEORAW_19','NEORAW_20','NEORAW_21','NEORAW_22','NEORAW_23','NEORAW_24','NEORAW_25','NEORAW_26','NEORAW_27','NEORAW_28','NEORAW_29','NEORAW_30','NEORAW_31','NEORAW_32','NEORAW_33','NEORAW_34','NEORAW_35','NEORAW_36','NEORAW_37','NEORAW_38','NEORAW_39','NEORAW_40','NEORAW_41','NEORAW_42','NEORAW_43','NEORAW_44','NEORAW_45','NEORAW_46','NEORAW_47','NEORAW_48','NEORAW_49','NEORAW_50','NEORAW_51','NEORAW_52','NEORAW_53','NEORAW_54','NEORAW_55','NEORAW_56','NEORAW_57','NEORAW_58','NEORAW_59','NEORAW_60','PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','IWRD_TOT','ListSort_Unadj','ER40_CR','LifeSatisf_Unadj','Endurance_Unadj','Dexterity_Unadj']

#column_tags = ['Strength_Unadj','Strength_AgeAdj','PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','IWRD_TOT','ListSort_Unadj','ER40_CR','LifeSatisf_Unadj','Endurance_Unadj','Dexterity_Unadj']

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

def train_linear(tag,train_input,train_output,test_input):
    #if 'WM' in tag:
    #    print(train_output)
    if tag == 'Gender' or tag == 'Age' or 'NEORAW_' in tag:
        if ARGS.opt == 'lin' or ARGS.opt == 'ransac'  or ARGS.opt=='lasso':
            if ARGS.opt == 'ransac' or ARGS.opt=='lasso':
                print('WARNING: RANSAC and Lasso is only for regression. Using logistic classifier instead')
            classifier = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=10000,penalty='none').fit(train_input,train_output)
            #classifier = LogisticRegression(random_state=0).fit(train_temp,train_outputs[:,pred_idx])
        elif ARGS.opt == 'nneigh':    
            classifier = KNeighborsClassifier().fit(train_input,train_output)
        elif ARGS.opt == 'boost':    
            classifier = GradientBoostingClassifier().fit(train_input,train_output)
        elif ARGS.opt == 'svm':    
            classifier = SVC().fit(train_input,train_output)
        elif ARGS.opt == 'mlp':    
            classifier = MLPClassifier(hidden_layer_sizes=(4,2),max_iter=10000,solver='lbfgs').fit(train_input,train_output)
        elif ARGS.opt == 'dummy':
            classifier = DummyClassifier(strategy='most_frequent').fit(train_input,train_output)
        #train_score = classifier.score(train_input,train_output)
        #test_score = classifier.score(test_input,test_output)
        train_pred = classifier.predict(train_input)
        test_pred = classifier.predict(test_input)
    else:
        if ARGS.opt == 'lin':
            classifier = LinearRegression(normalize=True).fit(train_input,train_output)
        elif ARGS.opt == 'lasso':
            classifier = Lasso(normalize=True,max_iter=10000).fit(train_input,train_output)
        elif ARGS.opt == 'ransac':
            classifier = RANSACRegressor().fit(train_input,train_output)
        elif ARGS.opt == 'nneigh':    
            classifier = KNeighborsRegressor().fit(train_input,train_output)
        elif ARGS.opt == 'boost':    
            classifier = GradientBoostingRegressor().fit(train_input,train_output)
        elif ARGS.opt == 'svm':    
            classifier = SVR().fit(train_input,train_output)
        elif ARGS.opt == 'mlp':    
            classifier = MLPRegressor(hidden_layer_sizes=(4,2),max_iter=1000,solver='lbfgs').fit(train_input,train_output)
        elif ARGS.opt == 'dummy':
            classifier = DummyRegressor(strategy='mean').fit(train_input,train_output)
        #train_score = classifier.score(train_input,train_output)
        #test_score = classifier.score(test_input,test_output)
        train_pred = classifier.predict(train_input)
        test_pred = classifier.predict(test_input)
    return train_pred,test_pred

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
    else:
        train_pred = np.zeros(len(train_out),dtype=str)
        test_pred = np.zeros(len(test_out),dtype=str)

    train_pred,test_pred = train_linear(pred_tag,train_in,train_out,test_in)
    #print(train_in.shape,test_in.shape)

    np.savez(os.path.join(pred_folder,'{}_tag{}_opt{}.npz'.format(ARGS.pred_file,pred_tag,ARGS.opt)),train_pred=train_pred[np.newaxis],train_true=train_out,test_pred=test_pred[np.newaxis],test_true=test_out,column_tags=pred_tag,row_tags='all',train_ids=train_ids,test_ids=test_ids)

