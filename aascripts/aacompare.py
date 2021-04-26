import numpy as np
import nibabel as nib
from aascripts.cliargs import get_args,get_parser
from autoatlas.analyze import overlap_coeff  
from autoatlas._utils import adjust_dims
import os
import csv
from .cliargs import HELP_MSG_DICT as HELP

def aacompare_parser(ret_dict=False):
    extra_args = {'train_list':[str,'File containing list of training samples.'],
                  'train_segvol':[str,'File containing segmentation volume.'],
                  'train_mask':[str,'Filepath of mask for training dataset.'],
                  'train_atlas':[str,'Fixed atlas file.'],
                  'train_olap_nnone':[str,'Overlap with no normalization.'],
                  'train_olap_nmin':[str,'Overlap normalized by the minimum number of samples.'],
                  'train_olap_nsum':[str,'Overlap normalized by the sum of samples.'],
                  'test_list':[str,'File containing list of testing samples.'],
                  'test_segvol':[str,'File containing segmentation volume.'],
                  'test_mask':[str,'Filepath of mask for testing dataset.'],
                  'test_atlas':[str,'Fixed atlas file.'],
                  'test_olap_nnone':[str,'Overlap with no normalization.'],
                  'test_olap_nmin':[str,'Overlap normalized by the minimum number of samples.'],
                  'test_olap_nsum':[str,'Overlap normalized by the sum of samples.']} 
    return get_parser(extra_args, ret_dict)

def write_csv(filen,data):
    assert data.ndim==2
    with open(filen,mode='w') as csv_file:
        csv_writer = csv.writer(csv_file,delimiter=',')
        csv_writer.writerow(['']+['FA{}'.format(k) for k in range(data.shape[1])])
        for i in range(data.shape[0]):
            temp = ['AA{}'.format(i)]+['{:.6e}'.format(k) for k in data[i].tolist()]
            csv_writer.writerow(temp)

def comp_vol(smpl_list,segvol_filen,mask_filen,atlas_filen,olap_nnone_filen,olap_nmin_filen,olap_nsum_filen):
    samples = []
    with open(smpl_list,'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            assert len(row)==1
            samples.append(row[0])

    for ID in samples:
        print(ID)
        autoa_ptr = nib.load(segvol_filen.format(ID))
        autoa_vol = autoa_ptr.get_fdata().astype(int)
        
        mask_ptr = nib.load(mask_filen.format(ID))
        mask_vol = mask_ptr.get_fdata().astype(bool)
        assert autoa_vol.shape==mask_vol.shape
        fixa_ptr = nib.load(atlas_filen.format(ID))
        fixa_vol = fixa_ptr.get_fdata().astype(int)
        assert fixa_vol.shape==autoa_vol.shape

        overlap_none = overlap_coeff(autoa_vol,fixa_vol,mask_vol,norm_type=None)
        overlap_min = overlap_coeff(autoa_vol,fixa_vol,mask_vol,norm_type='min')
        overlap_sum = overlap_coeff(autoa_vol,fixa_vol,mask_vol,norm_type='sum')
        
        write_csv(olap_nnone_filen.format(ID),np.squeeze(overlap_none))
        write_csv(olap_nmin_filen.format(ID),np.squeeze(overlap_min))
        write_csv(olap_nsum_filen.format(ID),np.squeeze(overlap_sum))
            
def main():
    ARGS = get_args(*aacompare_parser(ret_dict=True))

    comp_vol(ARGS['train_list'],ARGS['train_segvol'],ARGS['train_mask'],ARGS['train_atlas'],ARGS['train_olap_nnone'],ARGS['train_olap_nmin'],ARGS['train_olap_nsum'])
    comp_vol(ARGS['test_list'],ARGS['test_segvol'],ARGS['test_mask'],ARGS['test_atlas'],ARGS['test_olap_nnone'],ARGS['test_olap_nmin'],ARGS['test_olap_nsum'])
                        
