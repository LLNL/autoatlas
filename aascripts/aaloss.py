import argparse
from autoatlas.aatlas import AutoAtlas,partition_encode
from autoatlas.data import NibData
import os
import numpy as np
import nibabel as nib
from aascripts.cliargs import get_args,write_args,get_parser
from aascripts.cliargs import HELP_MSG_DICT as HELP
from aascripts.utils import get_dataset
import csv

def aaloss_parser(ret_dict=False):
    ARGS_dict = {'ckpt':[str,'File for storing run time data.'],
                'train_in':[str,'Filepath to input volume from training dataset.'],
                'train_mask':[str,'Filepath of mask for training dataset.'],
                'train_list':[str,'File containing list of training samples.'],
                'train_losses':[str,'CSV file to save training losses.'],
                'test_in':[str,'Filepath to input volume from testing dataset.'],
                'test_mask':[str,'Filepath of mask for testing dataset.'],
                'test_list':[str,'File containing list of testing samples.'],
                'test_losses':[str,'CSV file to save testing losses.'],
                'step_epochs':[int,'Compute loss every step_epochs number of epochs'],
                'in_dims':[str,'Dimensions separated by commas.'],
                'epochs':[int,'Number of epochs.']}
    return get_parser(ARGS_dict, ret_dict)

def make_dir(filen):
    folder = os.path.split(filen)
    assert len(folder)==2
    folder = folder[0]
    os.makedirs(folder,exist_ok=True)

def get_losses(autoseg,smpl_list,ndim,dvol_filen,dmask_filen):
    samples = []
    with open(smpl_list,'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            assert len(row)==1
            samples.append(row[0])
 
    batch = autoseg.ARGS['batch']
    tot_loss,mse_loss,smooth_loss,devr_loss,roi_loss = 0.0,0.0,0.0,0.0,0.0
    num_batches = 0
    for i in range(0,len(samples),batch):
        smpl_dataset,data_fin,mask_fin = get_dataset(samples[i:i+batch],ndim,dvol_filen,dmask_filen)
        smpl_tot,smpl_mse,smpl_smooth,smpl_devr,smpl_roi = autoseg.test(smpl_dataset)
        tot_loss += smpl_tot
        mse_loss += smpl_mse
        smooth_loss += smpl_smooth
        devr_loss += smpl_devr
        roi_loss += smpl_roi
        num_batches += 1  

    tot_loss /= num_batches
    mse_loss /= num_batches
    smooth_loss /= num_batches
    devr_loss /= num_batches
    roi_loss /= num_batches
    return tot_loss,mse_loss,smooth_loss,devr_loss,roi_loss
 
def main():
    ARGS = get_args(*aaloss_parser(ret_dict=True))
    
    cli_file = os.path.split(ARGS['cli_args'])[-1]
    write_args(ARGS,ARGS['cli_save'])
    ndim = len(ARGS['in_dims'].split(','))
 
    make_dir(ARGS['train_losses']) 
    train_file = open(ARGS['train_losses'],'w') 
    train_writer = csv.DictWriter(train_file,fieldnames=['Epoch','Tot Loss','RE Loss','NSS Loss','ADL Loss','ROI Loss'])
    train_writer.writeheader()
    make_dir(ARGS['test_losses']) 
    test_file = open(ARGS['test_losses'],'w',newline='')
    test_writer = csv.DictWriter(test_file,fieldnames=['Epoch','Tot Loss','RE Loss','NSS Loss','ADL Loss','ROI Loss'])
    test_writer.writeheader()

    for epoch in range(0,ARGS['epochs'],ARGS['step_epochs']): 
        autoseg = AutoAtlas(device='cuda',load_ckpt_epoch=epoch,ckpt_file=ARGS['ckpt'])
        if ARGS['train_list'] is not None:
            tot_loss,rel_loss,smooth_loss,devr_loss,roi_loss = get_losses(autoseg,ARGS['train_list'],ndim,ARGS['train_in'],ARGS['train_mask'])
            train_writer.writerow({'Epoch':epoch,'Tot Loss':tot_loss,'RE Loss':rel_loss,'NSS Loss':smooth_loss,'ADL Loss':devr_loss,'ROI Loss':roi_loss})
        if ARGS['test_list'] is not None:
            tot_loss,rel_loss,smooth_loss,devr_loss,roi_loss = get_losses(autoseg,ARGS['test_list'],ndim,ARGS['test_in'],ARGS['test_mask'])
            test_writer.writerow({'Epoch':epoch,'Tot Loss':tot_loss,'RE Loss':rel_loss,'NSS Loss':smooth_loss,'ADL Loss':devr_loss,'ROI Loss':roi_loss})

    train_file.close()
    test_file.close()

if __name__ == "__main__": 
    main() 
