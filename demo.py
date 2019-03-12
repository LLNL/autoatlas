from segmenter import AutoSegmenter
from data import SimDataset
from plot import stack_plot

#Parameters
num_epochs = 10
num_labels = 2
train_folder = '/data1/Data/TBI/simtrain64'
valid_folder = '/data1/Data/TBI/simvalid64'

#Datasets
train_data = SimDataset(train_folder,10)
valid_data = SimDataset(valid_folder,10)

#NN Model
autoseg = AutoSegmenter(num_labels,batch=4,eps=1e-8)

#Training
for epoch in range(num_epochs):
    print("Epoch {}".format(epoch))
    autoseg.train(train_data)
    autoseg.test(valid_data)
    tseg,tvol = autoseg.segment(train_data)
    vseg,vvol = autoseg.segment(valid_data)
    for i in range(num_labels):
        stack_plot([tvol[0,0],vvol[0,0]],[tseg[0,i],vseg[0,i]],'epoch_{}_label_{}_sample_1.png'.format(epoch,i))
        stack_plot([tvol[1,0],vvol[1,0]],[tseg[1,i],vseg[1,i]],'epoch_{}_label_{}_sample_2.png'.format(epoch,i))
