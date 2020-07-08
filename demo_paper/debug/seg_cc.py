import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import csv
import cc3d

tags = ['znvol_labs16_smooth0_1_devrr1_0_roir0']
#tags = ['mxvol_labs16_smooth0_005_devrr0_1_devrm0_9_emb16','mxvol_labs16_smooth0_005_devrr0_1_devrm0_9_emb8','mxvol_labs16_smooth0_005_devrr0_1_devrm0_9']
mode = 'train'
num_labels = [16,16,16]

smpl_list = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/{}_zn/subjects.txt'.format(mode)
#subjs = ['100307','100408','994273','995174']

def read_csv(filen):
    with open(filen,mode='r') as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        data = []
        for i,row in enumerate(csv_reader):
            if i!=0:
                data.append(np.array(row[1:],dtype=float))
    return np.stack(data,axis=0)

def overlay(T1,seg):
    img = np.stack((T1,T1,T1),axis=-1)/T1.max()
    labs = np.unique(seg)
    print(labs.min(),labs.max())
    for i,l in enumerate(labs):
        if i%3==0:
            rgb = [0,1,1]
        elif i%3==1:
            rgb = [1,0,1]
        elif i%3==2:
            rgb = [1,1,0]

        if l != 0:
            img[seg==l,0] = rgb[0]
            img[seg==l,1] = rgb[1]
            img[seg==l,2] = rgb[2]
    return img     

samples = []
with open(smpl_list,'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        assert len(row)==1
        samples.append(row[0])
samples = samples[0:1]

print(samples)
for idx,tag in enumerate(tags):
    print('Processing tag {}'.format(tag))
    data_max = []
    out_folder = os.path.join('figs',tag)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder) 
    
    for sub in samples:
        log_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/{}/{}_zn/{}/'.format(tag,mode,sub)
        seg = nib.load(os.path.join(log_dir,'seg_vol.nii.gz')).get_fdata()
        seg = seg.astype(np.uint32)

        log_dir = '/p/gpfs1/mohan3/Data/TBI/HCP/2mm/{}_zn/{}/'.format(mode,sub)
        mask = nib.load(os.path.join(log_dir,'mask.nii.gz')).get_fdata()
        mask = mask.astype(bool)
        T1 = nib.load(os.path.join(log_dir,'T1.nii.gz')).get_fdata()
        T1 = T1.astype(float)

        labs = cc3d.connected_components(seg)
        max_counts = []
        for l in range(num_labels[idx]):
            seg_labs = labs[np.bitwise_and(seg==l,mask)]
            uniq = np.unique(seg_labs)
            #print(uniq)
            counts = []  
            for u in uniq:
                counts.append(np.sum(seg_labs==u))
            if len(counts)==0:
                max_counts.append(np.nan)
            else:
                max_counts.append(max(counts)/sum(counts))
        print([float('{:.2f}'.format(cnt)) for cnt in max_counts])
        """    if max_counts[-1] < 0.9: 
                vol = labs.copy()
                vol[seg!=l] = 0
                vol[mask==False] = 0
                img = overlay(T1[48,:,:],vol[48,:,:])
                plt.imshow(img,cmap='gray')
                plt.colorbar()
                plt.show()
                img = overlay(T1[:,48,:],vol[:,48,:])
                plt.imshow(img,cmap='gray')
                plt.colorbar()
                plt.show()
                img = overlay(T1[:,:,48],vol[:,:,48])
                plt.imshow(img,cmap='gray')
                plt.colorbar()
                plt.show()"""
        #print([float('{:.2f}'.format(cnt)) for cnt in max_counts])
