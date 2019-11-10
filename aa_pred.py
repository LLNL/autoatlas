import numpy as np
import os
import csv
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.svm import SVC,SVR
from sklearn.neural_network import MLPClassifier,MLPRegressor

train_folder = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05/train_aa'
test_folder = '/p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05/test_aa'
gt_file = 'hcpdata/unrestricted_kaplan7_4_1_2019_18_31_31.csv' 

#column_tags = ['Gender','Strength_Unadj','FS_LCort_GM_Vol','FS_RCort_GM_Vol','FS_TotCort_GM_Vol','FS_SubCort_GM_Vol','FS_Total_GM_Vol','PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','IWRD_TOT','ListSort_Unadj','ER40_CR','LifeSatisf_Unadj','Endurance_Unadj','Dexterity_Unadj']
#column_tags = ['Strength_Unadj','MMSE_Score','PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','IWRD_TOT','ListSort_Unadj','ER40_CR','LifeSatisf_Unadj','Endurance_Unadj','Dexterity_Unadj']
column_tags = ['Strength_Unadj', 'Strength_AgeAdj']
"""'NEORAW_01', 'NEORAW_02',
       'NEORAW_03', 'NEORAW_04', 'NEORAW_05', 'NEORAW_06', 'NEORAW_07',
       'NEORAW_08', 'NEORAW_09', 'NEORAW_10', 'NEORAW_11', 'NEORAW_12',
       'NEORAW_13', 'NEORAW_14', 'NEORAW_15', 'NEORAW_16', 'NEORAW_17',
       'NEORAW_18', 'NEORAW_19', 'NEORAW_20', 'NEORAW_21', 'NEORAW_22',
       'NEORAW_23', 'NEORAW_24', 'NEORAW_25', 'NEORAW_26', 'NEORAW_27',
       'NEORAW_28', 'NEORAW_29', 'NEORAW_30', 'NEORAW_31', 'NEORAW_32',
       'NEORAW_33', 'NEORAW_34', 'NEORAW_35', 'NEORAW_36', 'NEORAW_37',
       'NEORAW_38', 'NEORAW_39', 'NEORAW_40', 'NEORAW_41', 'NEORAW_42',
       'NEORAW_43', 'NEORAW_44', 'NEORAW_45', 'NEORAW_46', 'NEORAW_47',
       'NEORAW_48', 'NEORAW_49', 'NEORAW_50', 'NEORAW_51', 'NEORAW_52',
       'NEORAW_53', 'NEORAW_54', 'NEORAW_55', 'NEORAW_56', 'NEORAW_57',
       'NEORAW_58', 'NEORAW_59', 'NEORAW_60', 'PicSeq_Unadj','CardSort_Unadj','Flanker_Unadj','PMAT24_A_CR','ReadEng_Unadj','PicVocab_Unadj','ProcSpeed_Unadj','VSPLOT_TC','IWRD_TOT','ListSort_Unadj','ER40_CR','LifeSatisf_Unadj','Endurance_Unadj','Dexterity_Unadj']"""

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
print(train_emb_codes)

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

def train_linear(tag,train_input,train_output,train_mask,test_input,test_output,test_mask):
    #if 'WM' in tag:
    #    print(train_output)
    if tag == 'Gender' or 'NEORAW_' in tag:
        #classifier = LogisticRegression(random_state=0,solver='lbfgs',max_iter=10000,penalty='none').fit(train_input[train_mask],train_output)
        #classifier = KNeighborsClassifier().fit(train_input[train_mask],train_output)
        classifier = GradientBoostingClassifier().fit(train_input[train_mask],train_output)
        #classifier = SVC().fit(train_input[train_mask],train_output)
        #classifier = MLPClassifier(hidden_layer_sizes=(64,16,4),max_iter=1000,solver='lbfgs').fit(train_input[train_mask],train_output)
        train_score = classifier.score(train_input[train_mask],train_output)
        test_score = classifier.score(test_input[test_mask],test_output)
        #classifier = LogisticRegression(random_state=0).fit(train_temp,train_outputs[:,pred_idx])
        if train_input.shape[1]>256:
            print(train_score,test_score)
    else:
        #classifier = LinearRegression().fit(train_input[train_mask],train_output)
        #classifier = KNeighborsRegressor().fit(train_input[train_mask],train_output)
        classifier = GradientBoostingRegressor().fit(train_input[train_mask],train_output)
        #classifier = SVR().fit(train_input[train_mask],train_output)
        #classifier = MLPRegressor(hidden_layer_sizes=(64,16,4),max_iter=1000,solver='lbfgs').fit(train_input[train_mask],train_output)
        train_score = classifier.score(train_input[train_mask],train_output)
        test_score = classifier.score(test_input[test_mask],test_output)
        if train_input.shape[1]>256:
            print(train_score,test_score)
    return train_score,test_score

print('--------- All seg and emb features -------------')
train_perf = np.zeros((len(row_tags),len(column_tags)),dtype=np.float32)
test_perf = np.zeros((len(row_tags),len(column_tags)),dtype=np.float32)
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
    if tag != 'Gender' and 'NEORAW_' not in tag:
        train_out = train_out.astype(float)    
        test_out = test_out.astype(float)    

    rid = 0
    train_in = np.concatenate((np.reshape(train_neigh_sims,(train_num,-1)),np.reshape(train_seg_probs,(train_num,-1)),np.reshape(train_emb_codes,(train_num,-1))),axis=-1)
    test_in = np.concatenate((np.reshape(test_neigh_sims,(test_num,-1)),np.reshape(test_seg_probs,(test_num,-1)),np.reshape(test_emb_codes,(test_num,-1))),axis=-1)
    train_perf[rid,cid],test_perf[rid,cid] = train_linear(tag,train_in,train_out,train_mask,test_in,test_out,test_mask)
    #print(train_in.shape,test_in.shape)

    rid = rid+1
    train_in = np.concatenate((np.reshape(train_neigh_sims,(train_num,-1)),np.reshape(train_seg_probs,(train_num,-1))),axis=-1)
    test_in = np.concatenate((np.reshape(test_neigh_sims,(test_num,-1)),np.reshape(test_seg_probs,(test_num,-1))),axis=-1)
    train_perf[rid,cid],test_perf[rid,cid] = train_linear(tag,train_in,train_out,train_mask,test_in,test_out,test_mask)
    #print(train_in.shape,test_in.shape)

    rid = rid+1
    train_in = np.reshape(train_emb_codes,(train_num,-1))
    test_in = np.reshape(test_emb_codes,(test_num,-1))
    train_perf[rid,cid],test_perf[rid,cid] = train_linear(tag,train_in,train_out,train_mask,test_in,test_out,test_mask)
    #print(train_in.shape,test_in.shape)

    for i in range(num_labels):
        rid = rid+1
        train_in = np.concatenate((np.reshape(train_neigh_sims[:,i],(train_num,-1)),np.reshape(train_seg_probs[:,i],(train_num,-1)),np.reshape(train_emb_codes[:,i],(train_num,-1))),axis=-1)
        test_in = np.concatenate((np.reshape(test_neigh_sims[:,i],(test_num,-1)),np.reshape(test_seg_probs[:,i],(test_num,-1)),np.reshape(test_emb_codes[:,i],(test_num,-1))),axis=-1)
        train_perf[rid,cid],test_perf[rid,cid] = train_linear(tag,train_in,train_out,train_mask,test_in,test_out,test_mask)
        #print(train_in.shape,test_in.shape)

        rid = rid+1
        train_in = np.concatenate((np.reshape(train_neigh_sims[:,i],(train_num,-1)),np.reshape(train_seg_probs[:,i],(train_num,-1))),axis=-1)
        test_in = np.concatenate((np.reshape(test_neigh_sims[:,i],(test_num,-1)),np.reshape(test_seg_probs[:,i],(test_num,-1))),axis=-1)
        train_perf[rid,cid],test_perf[rid,cid] = train_linear(tag,train_in,train_out,train_mask,test_in,test_out,test_mask)
        #print(train_in.shape,test_in.shape)

        rid = rid+1
        train_in = np.reshape(train_emb_codes[:,i],(train_num,-1))
        test_in = np.reshape(test_emb_codes[:,i],(test_num,-1))
        train_perf[rid,cid],test_perf[rid,cid] = train_linear(tag,train_in,train_out,train_mask,test_in,test_out,test_mask)
        #print(train_in.shape,test_in.shape)
        #mcls = 'M' if np.sum(train_outputs[:,0]=='M')>np.sum(train_outputs[:,0]=='F') else 'F'
        #print('Test majority',np.mean(test_outputs[:,0]==mcls))

np.savez('aa4_boost_perfs.npz',train_perf=train_perf,test_perf=test_perf,row_tags=row_tags,column_tags=column_tags)
print(test_perf[0,:])
