import argparse
from autoatlas.aatlas import AutoAtlas,partition_encode
from autoatlas.data import NibData
import os
import numpy as np
import nibabel as nib
from aascripts.cliargs import get_parser,get_args,write_args
from aascripts.cliargs import HELP_MSG_DICT as HELP
from aascripts.utils import get_dataset
import csv
   
def aainfer_parser(ret_dict=False): 
    ARGS_dict = {'ckpt':[str,'File for storing run time data.'],
                'in_dims':[str,'Dimensions separated by comma.'],
                'train_in':[str,'Filepath to input volume from training dataset.'],
                'train_mask':[str,'Filepath of mask for training dataset.'],
                'train_list':[str,'File containing list of training samples.'],
                'train_segcode':[str,'File to store segmentation encoding of volume and surface area.'],
                'train_embcode':[str,'File to store embedding for each sample.'],
                'train_allcode':[str,'File to store segmentation and embedding data for each sample.'],
                'train_probvol':[str,'File to store probability volumes.'],
                'train_segvol':[str,'File to store segmentation volumes.'],
                'train_recvol':[str,'File to store reconstruction.'],
                'test_in':[str,'Filepath to input volume from testing dataset.'],
                'test_mask':[str,'Filepath of mask for testing dataset.'],
                'test_list':[str,'File containing list of testing samples.'],
                'test_segcode':[str,'File to store segmentation encoding of volume and surface area.'],
                'test_embcode':[str,'File to store embedding for each sample.'],
                'test_allcode':[str,'File to store segmentation and embedding data for each sample.'],
                'test_probvol':[str,'File to store probability volumes.'],
                'test_segvol':[str,'File to store segmentation volumes.'],
                'test_recvol':[str,'File to store reconstruction.'],
                'load_epoch':[int,'Model epoch to load. If negative, does not load model.']}
    return get_parser(ARGS_dict, ret_dict)

def get_meta(filen):
    dptr = nib.load(filen)
    return dptr.get_affine(),dptr.get_header()

def save_nifti(filen,data,affine,header):
    folder = os.path.split(filen)
    assert len(folder)==2
    folder = folder[0]
    os.makedirs(folder,exist_ok=True)
    
    header.set_data_dtype(data.dtype) 
    header.set_slope_inter(None,None)
    dptr = nib.Nifti1Image(data,affine,header)
    dptr.to_filename(filen)

def write_code(filen,code):
    folder = os.path.split(filen)
    assert len(folder)==2
    folder = folder[0]
    os.makedirs(folder,exist_ok=True)
    
    with open(filen,mode='w') as csv_file:
        csv_writer = csv.writer(csv_file,delimiter=',')
        csv_writer.writerow(['']+['f{}'.format(i) for i in range(code.shape[1])])
        for i in range(code.shape[0]):
            data = ['fv{}'.format(i)]+['{:.6e}'.format(c) for c in code[i]]
            csv_writer.writerow(data)

def process_data(autoseg,ndim,smpl_list,dvol_filen,dmask_filen,segc_filen,embc_filen,allc_filen,probv_filen,segv_filen,recc_filen):
    samples = []
    with open(smpl_list,'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            assert len(row)==1
            samples.append(row[0])
 
    batch = autoseg.ARGS['batch']
    for i in range(0,len(samples),batch):
        smpl_dataset,data_fin,mask_fin = get_dataset(samples[i:i+batch],ndim,dvol_filen,dmask_filen)
        segs,recs,masks,embeds,data_fout,mask_fout = autoseg.process(smpl_dataset,ret_input=False)
        assert data_fin==data_fout and len(data_fin)<=batch
        assert mask_fin==mask_fout and len(mask_fin)<=batch
        for j in range(len(data_fout)):
            smpl = samples[i+j]
            print(smpl)
            vol_meas,area_meas = partition_encode(segs[j],masks[j],ndim)
            code = np.stack((vol_meas,area_meas),axis=1)
            write_code(segc_filen.format(smpl),code) 
            write_code(embc_filen.format(smpl),embeds[j]) 
            write_code(allc_filen.format(smpl),np.concatenate((code,embeds[j]),axis=1)) 

            if data_fout[j][-7:]=='.nii.gz' or data_fout[j][-4:]=='.nii':
                aff,head = get_meta(data_fout[j])
                axes = (1,2,3,0) if segs[j].ndim==4 else (1,2,0)
                arr = np.transpose(segs[j].astype(np.float32,order='C'),axes=axes)
                save_nifti(probv_filen.format(smpl),arr,aff,head)
                arr = np.argmax(arr,axis=-1)
                save_nifti(segv_filen.format(smpl),arr,aff,head)
                axes = (1,2,3,0) if recs[j].ndim==4 else (1,2,0)
                arr = np.transpose(recs[j].astype(np.float32,order='C'),axes=axes) 
                save_nifti(recc_filen.format(smpl),arr,aff,head)
            else:
                axes = (1,2,3,0) if segs[j].ndim==4 else (1,2,0)
                arr = np.transpose(segs[j].astype(np.float32,order='C'),axes=axes)
                np.save(probv_filen.format(smpl),arr) 
                arr = np.argmax(arr,axis=-1)
                np.save(segv_filen.format(smpl),arr) 
                axes = (1,2,3,0) if recs[j].ndim==4 else (1,2,0)
                arr = np.transpose(recs[j].astype(np.float32,order='C'),axes=axes) 
                np.save(recc_filen.format(smpl),arr) 
 
def main():
    ARGS = get_args(*aainfer_parser(ret_dict=True))
    cli_file = os.path.split(ARGS['cli_args'])[-1]
    write_args(ARGS,ARGS['cli_save'])
    ndim = len(ARGS['in_dims'].split(','))
    
    if ARGS['load_epoch'] >= 0:
        autoseg = AutoAtlas(device='cuda',load_ckpt_epoch=ARGS['load_epoch'],ckpt_file=ARGS['ckpt'])
    else:
        raise ValueError('load_epoch must be specified')
    
    if ARGS['train_list'] is not None:
        process_data(autoseg,ndim,ARGS['train_list'],ARGS['train_in'],ARGS['train_mask'],ARGS['train_segcode'],ARGS['train_embcode'],ARGS['train_allcode'],ARGS['train_probvol'],ARGS['train_segvol'],ARGS['train_recvol'])

    if ARGS['test_list'] is not None:
        process_data(autoseg,ndim,ARGS['test_list'],ARGS['test_in'],ARGS['test_mask'],ARGS['test_segcode'],ARGS['test_embcode'],ARGS['test_allcode'],ARGS['test_probvol'],ARGS['test_segvol'],ARGS['test_recvol'])

if __name__ == "__main__": 
    main() 
