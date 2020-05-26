from autoatlas import AutoAtlas
from autoatlas.data import NibData
import os
import csv
from .cliargs import get_args,write_args
from .cliargs import HELP_MSG_DICT as HELP
from .utils import get_dataset,get_samples
import numpy as np

def main():
    ARGS_dict = {
            'train_in':[str,'Filename of input volumes for training autoatlas.'],
            'train_mask':[str,'Filename of masks for input volumes in training set.'],
            'train_list':[str,'File containing list of training samples.'],
            'test_in':[str,'Filename of input volumes for testing autoatlas.'],
            'test_mask':[str,'Filename of masks for input volumes in test set.'],
            'test_list':[str,'File containing list of test samples.'],
            'ckpt':[str,'File for storing run time data.'],
            'epochs':[int,'Number of epochs.'],
            'batch':[int,'Batch size.'],
            'lr':[float,'Learning rate.'],
            'distr':[bool,'If used, distribute model among all GPUs in a node.'],
            'load_epoch':[int,'Model epoch to load. If negative, does not load model.'],
            'num_labels':[int,'Number of class labels.'],
            'in_chan':[int,'Number of channels in input image or volume.'],
            'in_dims':[str,'Dimensions separated by comma.'],
            'unet_chan':[int,'Number of U-net input channels.'],
            'unet_blocks':[int,'Number of U-net blocks each with two conv layers.'],
            'aenc_chan':[int,'Number of autoencoder channels.'],
            'aenc_depth':[int,'Depth of autoencoder.'],
            'in_chan':[int,'Number of channels in input image or volume.'],
            're_pow':[int,'Power (of norm) for absolute reconstruction error.'],
            'rel_reg':[float,'Weight for REL loss.'],
            'smooth_reg':[float,'Regularization enforcing smoothness.'],
            'devr_reg':[float,'Regularization parameter for anti-devouring loss.'],
            'roi_reg':[float,'Regularization parameter for ROI loss.'],
            'roi_rad_mult':[float,'Radius parameter for ROI loss.'],
            'min_freqs_mult':[float,'Minimum probability of occurance of each class'],
            }

    ARGS = get_args(ARGS_dict)
    if not os.path.exists(ARGS['cli_args']):
        raise ValueError('{} not found'.format(ARGS['cli_args']))
        
    write_args(ARGS,ARGS['cli_save'])

    #Datasets
    samples = get_samples(ARGS['train_list']) 
    train_data,_,_ = get_dataset(samples,ARGS['train_in'],ARGS['train_mask'])
    if ARGS['test_list'] is not None:
        samples = get_samples(ARGS['test_list']) 
        test_data,_,_ = get_dataset(samples,ARGS['test_in'],ARGS['test_mask'])

    #NN Model
    dims = [int(d) for d in ARGS['in_dims'].split(',')]
    min_freqs = ARGS['min_freqs_mult']/ARGS['num_labels']

    if len(dims)==3:
        roi_rad = ((3.0*dims[0]*dims[1]*dims[2])/(4.0*np.pi*ARGS['num_labels']))**(1.0/3)
    elif len(dims)==2:
        roi_rad = ((dims[0]*dims[1])/(np.pi*ARGS['num_labels']))**(1.0/2)
    else:
        raise ValueError('Only 3D and 2D inputs are supported.')
    roi_rad = ARGS['roi_rad_mult']*roi_rad
    print('min_freqs = {}, roi_rad = {}'.format(min_freqs,roi_rad))

    if ARGS['load_epoch'] >= 0:
        autoseg = AutoAtlas(ARGS['num_labels'],sizes=dims,data_chan=ARGS['in_chan'],rel_reg=ARGS['rel_reg'],smooth_reg=ARGS['smooth_reg'],devr_reg=ARGS['devr_reg'],roi_reg=ARGS['roi_reg'],min_freqs=min_freqs,roi_rad=roi_rad,batch=ARGS['batch'],lr=ARGS['lr'],unet_chan=ARGS['unet_chan'],unet_blocks=ARGS['unet_blocks'],aenc_chan=ARGS['aenc_chan'],aenc_depth=ARGS['aenc_depth'],re_pow=ARGS['re_pow'],distr=ARGS['distr'],device='cuda',load_ckpt_epoch=ARGS['load_epoch'],ckpt_file=ARGS['ckpt'])
    elif ARGS['load_epoch'] == -1:
        autoseg = AutoAtlas(ARGS['num_labels'],sizes=dims,data_chan=ARGS['in_chan'],rel_reg=ARGS['rel_reg'],smooth_reg=ARGS['smooth_reg'],devr_reg=ARGS['devr_reg'],roi_reg=ARGS['roi_reg'],min_freqs=min_freqs,roi_rad=roi_rad,batch=ARGS['batch'],lr=ARGS['lr'],unet_chan=ARGS['unet_chan'],unet_blocks=ARGS['unet_blocks'],aenc_chan=ARGS['aenc_chan'],aenc_depth=ARGS['aenc_depth'],re_pow=ARGS['re_pow'],distr=ARGS['distr'],device='cuda',ckpt_file=ARGS['ckpt'])
    else:
        raise ValueError('load_epoch must be greater than or equal to -1')

    #Training
    prev_epoch = ARGS['load_epoch']
    for epoch in range(prev_epoch+1,ARGS['epochs']):
        print("Epoch {}".format(epoch))
        autoseg.train(train_data)
        if ARGS['test_list'] is not None:
            autoseg.test(test_data)
        autoseg.ckpt(epoch,ARGS['ckpt'])
        ARGS['load_epoch'] = epoch

#End of function main()

if __name__ == "__main__": 
    main() 
