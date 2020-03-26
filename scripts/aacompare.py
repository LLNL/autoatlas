import numpy as np
import nibabel as nib
from .cliargs import get_args
from autoatlas.analyze import overlap_coeff  
from autoatlas.utils import adjust_dims
import os
import csv

def save_nifti(filen,data,affine,header):
    header.set_data_dtype(data.dtype) 
    header.set_slope_inter(None,None)
    dptr = nib.Nifti1Image(data,affine,header)
    dptr.to_filename(filen)

def write_csv(savedir,filen,data):
    assert data.ndim==2
    filen = os.path.join(savedir,filen)
    with open(filen,mode='w') as csv_file:
        csv_writer = csv.writer(csv_file,delimiter=',')
        csv_writer.writerow(['']+['FA{}'.format(k) for k in range(data.shape[1])])
        for i in range(data.shape[0]):
            temp = ['AA{}'.format(i)]+['{:.6e}'.format(k) for k in data[i].tolist()]
            csv_writer.writerow(temp)

def comp_vol(savedir,atlasdir,dims):
    autoa_files = [f for f in os.listdir(savedir) if '_T1w_brain_2_aaparts' in f] 
    if savedir is not None:
        for f in autoa_files:
            ID = os.path.split(f)[-1].split('_')[0]
            autoa_ptr = nib.load(os.path.join(savedir,f))
            autoa_vol = autoa_ptr.get_fdata().astype(int)
            assert autoa_vol.shape==tuple(dims)

            mask_ptr = nib.load(os.path.join(savedir,'{}_T1w_brain_2_aamask.nii.gz'.format(ID)))
            mask_vol = mask_ptr.get_fdata().astype(bool)
            assert mask_vol.shape==tuple(dims)

            fixa_file = os.path.join(atlasdir,'{}-tissue_2mm.nii.gz'.format(ID))
            fixa_ptr = nib.load(fixa_file)
            fixa_vol = fixa_ptr.get_fdata().astype(int)
            fixa_vol = adjust_dims(fixa_vol,dims)
            save_nifti(os.path.join(savedir,'{}-tissue_2mm.nii.gz'.format(ID)),fixa_vol,autoa_ptr.get_affine(),autoa_ptr.get_header())            

            overlap_none = overlap_coeff(autoa_vol,fixa_vol,mask_vol,norm_type=None)
            overlap_min = overlap_coeff(autoa_vol,fixa_vol,mask_vol,norm_type='min')
            overlap_sum = overlap_coeff(autoa_vol,fixa_vol,mask_vol,norm_type='sum')
            
            write_csv(savedir,'{}_overlap.csv'.format(ID),np.squeeze(overlap_none))
            write_csv(savedir,'{}_overlap_minnorm.csv'.format(ID),np.squeeze(overlap_min))
            write_csv(savedir,'{}_overlap_sumnorm.csv'.format(ID),np.squeeze(overlap_sum))
            
def main():
    extra_args = {'atlas_dir':[str,'Directory where atlas volumes are stored. Voxel values of the atlas must be of integer type.']}
    
    ARGS = get_args(extra_args)
    dims = [ARGS['size_dim'] for _ in range(ARGS['space_dim'])]
    comp_vol(ARGS['train_savedir'],ARGS['atlas_dir'],dims)
    comp_vol(ARGS['test_savedir'],ARGS['atlas_dir'],dims)
                        
