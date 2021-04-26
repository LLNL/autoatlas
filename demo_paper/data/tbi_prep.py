import numpy as np
import nibabel as nib
import os
import csv
from skimage.filters import threshold_otsu
from skimage.util import montage
import cc3d

np.random.seed(0)

mri_folder = '/p/lustre2/kaplan7/tbi_t1_resample/2mm'
save_folder = '/p/lustre1/mohan3/Data/TBI/HCP/2mm/tbi_mx'
dims = [96,96,96]
#stdev = 286.318
#mean = 698.45
#stddev = 165.27 
maxm = 1116.786

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

mri_files = np.array([f for f in os.listdir(mri_folder) if '.nii.gz' in f])
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
        ID = in_filen.split('.')[0][6:]
        print(ID)
        IDs.append(ID)
        out_ftemp = os.path.join(out_folder,'{}'.format(ID))
        os.makedirs(out_ftemp)

        vol,aff,head = get_data(os.path.join(mri_folder,in_filen))
        t1sh = vol.shape
        vol = adjust_dims(vol)
        thresh = threshold_otsu(montage(vol,grid_shape=(1,vol.shape[0])))
        mask = (vol < thresh/1.5)
 
        labs = cc3d.connected_components(mask)
        uniqs = np.unique(labs)
        max_counts,max_label = 0,0
        for u in uniqs:
            cnt = np.sum(np.bitwise_and(labs==u,mask==True))
            if cnt > max_counts:
                max_counts = cnt
                max_label = u
        mask = np.bitwise_not(labs==max_label).astype(np.uint8) 
        out_filen = os.path.join(out_ftemp,'mask.nii.gz'.format(ID)) 
        save_nifti(out_filen,mask,aff,head)

        #mean,stddev = np.mean(vol[mask==1.0]),np.std(vol[mask==1.0])       
        #vol = (vol-mean)/stddev
        vol = vol/maxm
        out_filen = os.path.join(out_ftemp,'T1.nii.gz'.format(ID))
        save_nifti(out_filen,vol.astype(np.float32),aff,head)
        
    with open(os.path.join(out_folder,'subjects.txt'),'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows([[ID] for ID in IDs])

save_orgdata(mri_files,save_folder)


