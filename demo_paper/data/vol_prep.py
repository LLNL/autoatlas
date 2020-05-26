import numpy as np
import nibabel as nib
import os
import csv

np.random.seed(0)

mri_folder = '/p/lustre1/hcpdata/processed/T1_decimate/1mm'
tt_folder = '/p/lustre1/hcpdata/processed/5TT/seg_flip/1mm'
train_folder = '/p/lustre1/mohan3/Data/TBI/HCP/1mm/train'
test_folder = '/p/lustre1/mohan3/Data/TBI/HCP/1mm/test'
dims = [192,192,192]
train_num = 80 #percent
stdev = 286.318

if not os.path.exists(train_folder):
    os.makedirs(train_folder)

if not os.path.exists(test_folder):
    os.makedirs(test_folder)

mri_files = np.array(os.listdir(mri_folder))
train_num = int(np.round(len(mri_files)*train_num/100))

perm_idx = np.random.permutation(np.arange(0,len(mri_files),dtype=int))
train_files = mri_files[perm_idx[:train_num]]
test_files = mri_files[perm_idx[train_num:]]

dims = np.array(dims,dtype=int)
def adjust_dims(vol):
    iB = (np.array(vol.shape,dtype=int)-dims)//2
    iE = dims+iB
    if iB[0]>=0: 
        vol = vol[iB[0]:iE[0]]
    if iB[1]>=0: 
        vol = vol[:,iB[1]:iE[1]]
    if iB[2]>=0: 
        vol = vol[:,:,iB[2]:iE[2]]
    sh = np.array(vol.shape,dtype=int)
    wid = dims-sh
    wid[wid<0] = 0
    return np.pad(vol,((0,wid[0]),(0,wid[1]),(0,wid[2])),mode='constant')

def get_data(filen):
    dptr = nib.load(filen)
    return dptr.get_fdata(),dptr.get_affine(),dptr.get_header()

def save_nifti(filen,data,affine,header):
    header.set_data_dtype(data.dtype) 
    header.set_slope_inter(None,None)
    dptr = nib.Nifti1Image(data,affine,header)
    dptr.to_filename(filen)

def save_orgdata(in_files,out_folder):
    IDs = []
    for in_filen in in_files:
        ID = int(in_filen.split('_')[0])
        print(ID)
        IDs.append(ID)
        out_ftemp = os.path.join(out_folder,'{}'.format(ID))
        os.makedirs(out_ftemp)

        vol,aff,head = get_data(os.path.join(mri_folder,in_filen))
        t1sh = vol.shape
        vol = adjust_dims(vol)/stdev
        out_filen = os.path.join(out_ftemp,'T1.nii.gz'.format(ID))
        save_nifti(out_filen,vol.astype(np.float32),aff,head)
       
        vol,aff,head = get_data(os.path.join(tt_folder,'{}-tissue_1mm.nii.gz'.format(ID)))
        assert vol.shape==t1sh

        mask = adjust_dims((vol>0).astype(np.uint8))
        tt = adjust_dims((vol-1).astype(int))
        tt[tt<0] = 0
        assert np.any(tt<=3)

        out_filen = os.path.join(out_ftemp,'mask.nii.gz'.format(ID)) 
        save_nifti(out_filen,mask,aff,head)
        out_filen = os.path.join(out_ftemp,'5TT.nii.gz'.format(ID)) 
        save_nifti(out_filen,tt,aff,head)

    with open(os.path.join(out_folder,'subjects.txt'),'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows([[ID] for ID in IDs])

save_orgdata(train_files,train_folder)
save_orgdata(test_files,test_folder)


