import numpy as np
import nibabel as nib
import os
import csv
from skimage.filters import threshold_otsu
from skimage.util import montage
import cc3d
import matplotlib.pyplot as plt

np.random.seed(0)

mri_folder = '/p/lustre2/kaplan7/tbi_t1_resample/2mm'
save_folder = '/p/lustre1/mohan3/Data/TBI/HCP/2mm/test_tbi_znrt'
dims = [96,96,96]
#stdev = 286.318
#mean = 698.45
#stddev = 165.27 
#maxm = 1116.786

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

mri_files = np.array([f for f in os.listdir(mri_folder) if '.nii.gz' in f])
dims = np.array(dims,dtype=int)
def adjust_dims(vol,z_min=None,z_max=None,y_min=None,y_max=None,x_min=None,x_max=None):
    if z_min is None or z_max is None:
        vol_zsum = np.nonzero(np.sum(vol,axis=(1,2)))
        z_min,z_max = np.min(vol_zsum),np.max(vol_zsum) 

    if y_min is None or y_max is None:
        vol_ysum = np.nonzero(np.sum(vol,axis=(0,2)))
        y_min,y_max = np.min(vol_ysum),np.max(vol_ysum) 

    if x_min is None or x_max is None:
        vol_xsum = np.nonzero(np.sum(vol,axis=(0,1)))
        x_min,x_max = np.min(vol_xsum),np.max(vol_xsum) 

    assert z_min>=0 and y_min>=0 and x_min>=0
    vol = np.roll(vol,-z_min,axis=0)
    vol = np.roll(vol,-y_min,axis=1)
    vol = np.roll(vol,-x_min,axis=2)

    if vol.shape[0] > dims[0]:
        vol = vol[:dims[0]]  
    else:
        vol = np.pad(vol,pad_width=((0,dims[0]-vol.shape[0]),(0,0),(0,0)),mode='edge')
    
    if vol.shape[1] > dims[1]:
        vol = vol[:,:dims[1]]  
    else:
        vol = np.pad(vol,pad_width=((0,0),(0,dims[1]-vol.shape[1]),(0,0)),mode='edge')
    
    if vol.shape[2] > dims[2]:
        vol = vol[:,:,:dims[2]]  
    else:
        vol = np.pad(vol,pad_width=((0,0),(0,0),(0,dims[2]-vol.shape[2])),mode='edge')
    
    #plt.imshow(vol[dims[0]//2])
    #plt.colorbar()
    #plt.show() 
    #plt.imshow(vol[:,dims[1]//2])
    #plt.colorbar()
    #plt.show() 
    #plt.imshow(vol[:,:,dims[2]//2])
    #plt.colorbar()
    #plt.show()

    shift_z = (dims[0]-z_max+z_min)//2 
    shift_y = (dims[1]-y_max+y_min)//2 
    shift_x = (dims[2]-x_max+x_min)//2
    
    if shift_z<0 or shift_y<0 or shift_x<0:
        print('WARN: shift_z = {}, shift_y = {}, shift_x = {}'.format(shift_z,shift_y,shift_x))
 
    vol = np.roll(vol,shift_z,axis=0)
    vol = np.roll(vol,shift_y,axis=1)
    vol = np.roll(vol,shift_x,axis=2)
   
    #plt.imshow(vol[dims[0]//2])
    #plt.colorbar()
    #plt.show() 
    #plt.imshow(vol[:,dims[1]//2])
    #plt.colorbar()
    #plt.show() 
    #plt.imshow(vol[:,:,dims[2]//2])
    #plt.colorbar()
    #plt.show() 
    return vol,(z_min,z_max),(y_min,y_max),(x_min,x_max)

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
        mask,(z_min,z_max),(y_min,y_max),(x_min,x_max) = adjust_dims(mask) 
        out_filen = os.path.join(out_ftemp,'mask.nii.gz'.format(ID)) 
        save_nifti(out_filen,mask,aff,head)

        vol,_,_,_ = adjust_dims(vol,z_min,z_max,y_min,y_max,x_min,x_max)
        assert vol.shape[0]==dims[0] and vol.shape[1]==dims[1] and vol.shape[2]==dims[2]
        mean,stddev = np.mean(vol[mask==1.0]),np.std(vol[mask==1.0])       
        vol = (vol-mean)/stddev
        #vol = vol/maxm
        print('Old Mean={},StdDev={}'.format(mean,stddev))
        print('New Mean={},StdDev={}'.format(np.mean(vol[mask==1.0]),np.std(vol[mask==1.0])))
        #plt.imshow(vol[dims[0]//2])
        #plt.colorbar()
        #plt.show() 
        #plt.imshow(vol[:,dims[1]//2])
        #plt.colorbar()
        #plt.show() 
        #plt.imshow(vol[:,:,dims[2]//2])
        #plt.colorbar()
        #plt.show() 
        out_filen = os.path.join(out_ftemp,'T1.nii.gz'.format(ID))
        save_nifti(out_filen,vol.astype(np.float32),aff,head)
        
    with open(os.path.join(out_folder,'subjects.txt'),'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows([[ID] for ID in IDs])

save_orgdata(mri_files,save_folder)
