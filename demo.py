from segmenter import AutoSegmenter
from data import SimDataset
from plot import stack_plot
import os

#Parameters
num_epochs = 50
num_labels = 3
train_folder = '/data1/Data/TBI/simtrain64'
valid_folder = '/data1/Data/TBI/simvalid64'

#Datasets
train_files = [os.path.join(train_folder,f) for f in os.listdir(train_folder) if f[-3:]=='.h5']
valid_files = [os.path.join(valid_folder,f) for f in os.listdir(valid_folder) if f[-3:]=='.h5']

train_data = SimDataset(train_files)
valid_data = SimDataset(valid_files[:10])

train_seg = SimDataset(train_files[0])
valid_seg = SimDataset(valid_files[0])

#NN Model
autoseg = AutoSegmenter(num_labels,batch=8,eps=1e-15,lr=1e-4,device='cuda')

#Training
for epoch in range(num_epochs):
    print("Epoch {}".format(epoch))
    autoseg.train(train_data)
    autoseg.test(valid_data)
    tseg,tvol = autoseg.segment(train_seg)
    vseg,vvol = autoseg.segment(valid_seg)
    autoseg.checkpoint(epoch)
    for i in range(num_labels):
        stack_plot([tvol[0,0],vvol[0,0]],[tseg[0,i],vseg[0,i]],'epoch_{}_label_{}_sample_1.png'.format(epoch,i))
