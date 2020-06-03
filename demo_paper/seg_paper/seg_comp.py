import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import csv

#tags = ['aa_freqs0_05','aa_smooth0_2_freqs0_05','aa_devr0_1','aa_smooth0_2_devr0_1']
tags = ['aa_labs16_smooth0_1_devrm0_8_roim1_2_lb3']
mode = 'train'
num_labels = 16

smpl_list = '/p/lustre1/mohan3/Data/TBI/HCP/2mm/{}/subjects.txt'.format(mode)
#subjs = ['100307','100408','994273','995174']

def read_csv(filen):
    with open(filen,mode='r') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        data = []
        for i,row in enumerate(csv_reader):
            if i!=0:
                data.append(np.array(row[1:],dtype=float))
    return np.stack(data,axis=0)

samples = []
with open(smpl_list,'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row)==1
        samples.append(row[0])

for tag in tags:
    print('Processing tag {}'.format(tag))
    data_max = []
    out_folder = os.path.join('figs',tag)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder) 
    
    for sub in samples:
        log_dir = '/p/lustre1/mohan3/Data/TBI/HCP/2mm/{}/{}/{}/'.format(tag,mode,sub)

        data = read_csv(os.path.join(log_dir,'olap_nmin.csv'))
        data_max.append(data.max(axis=1))
        #data_max = ['{}'.format(sub)]+['{:.2f}'.format(d) for d in data_max]
        #print(','.join(data_max))
        #print('{},min={:.2f},max={:.2f},mean={:.2f},median={:.2f}'.format(sub,data_max.min(),data_max.max(),data_max.mean(),np.median(data_max)))
        
    data_max = np.stack(data_max,axis=1)
    for i in range(data_max.shape[0]):
        plt.hist(data_max[i],bins=100,density=True,range=(0,1))
        plt.savefig(os.path.join(out_folder,'olap_{}_r{}.png'.format(mode,i)))
        plt.close() 
