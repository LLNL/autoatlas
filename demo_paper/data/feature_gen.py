import numpy as np
import csv
import os

train_folder = '/p/lustre1/mohan3/Data/TBI/HCP/2mm/train_nm'
test_folder = '/p/lustre1/mohan3/Data/TBI/HCP/2mm/test_nm'

def read_scolcsv(filen):
    data = []
    with open(filen) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            assert len(row)==1
            data.append(row[0])
    return data 

fture = read_scolcsv('feature_list.csv')
targ = read_scolcsv('target_list.csv')

fture_dict,targ_dict = {},{}
with open('unrestricted_kaplan7_4_1_2019_18_31_31.csv',mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for i,row in enumerate(csv_reader):
        row = np.array(row)
        if i==0:
            fture_idx = [j for j in range(len(row)) if row[j] in fture]
            targ_idx = [j for j in range(len(row)) if row[j] in targ]
            fture_labs = row[fture_idx]
            targ_labs = row[targ_idx]
        else:
            #if np.all(row[fture_idx]!='') and np.all(row[targ_idx]!=''):
            fture_dict.update({str(row[0]):row[fture_idx]})        
            targ_dict.update({str(row[0]):row[targ_idx]})        
       
train_subj = read_scolcsv(os.path.join(train_folder,'subjects.txt'))
test_subj = read_scolcsv(os.path.join(test_folder,'subjects.txt'))

def write_2colcsv(folder,subj):
    filt_subj = []
    for s in subj:
        print(s)
        rows = [['name','value']]
        for i,lab in enumerate(fture_labs):
            rows.append([lab,fture_dict[s][i]])
       
        sfder = os.path.join(folder,s)
        with open(os.path.join(sfder,'feature.csv'),'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(rows)
        
        rows = [['name','value']]
        for i,lab in enumerate(targ_labs):
            rows.append([lab,targ_dict[s][i]])
            
        with open(os.path.join(sfder,'target.csv'),'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(rows)

        if np.all(fture_dict[s]!='') and np.all(targ_dict[s]!=''):
            filt_subj.append([s])
        else:
            print('Subj {} has missing data: features = {}, targets = {}'.format(s,fture_dict[s],targ_dict[s]))

    with open(os.path.join(folder,'subjects_rlearn.txt'),'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(filt_subj) 

write_2colcsv(train_folder,train_subj)
write_2colcsv(test_folder,test_subj)
