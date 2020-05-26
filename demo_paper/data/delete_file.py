import numpy as np
import csv
import os

train_folder = '/p/lustre1/mohan3/Data/TBI/HCP/2mm/train'
test_folder = '/p/lustre1/mohan3/Data/TBI/HCP/2mm/test'

tags = []

def read_scolcsv(filen):
    data = []
    with open(filen) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            assert len(row)==1
            data.append(row[0])
    return data 

train_subj = read_scolcsv(os.path.join(train_folder,'subjects.txt'))
test_subj = read_scolcsv(os.path.join(test_folder,'subjects.txt'))

def del_file(folder,subj,filen):
    for s in subj:
        sfder = os.path.join(folder,s)
        os.remove(os.path.join(sfder,filen))

del_file(train_folder,train_subj,'meta_data.csv')
del_file(test_folder,test_subj,'meta_data.csv')
del_file(train_folder,train_subj,'meta.csv')
del_file(test_folder,test_subj,'meta.csv')
