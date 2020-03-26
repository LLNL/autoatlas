import argparse
import configparser as cp
import os

def config_fetch(label,cfg_obj,cfg_file,dtype,parser,raise_ex=True):   
    try:
        if dtype is int:
            cfg_arg = cfg_obj.getint('command_line_arguments',label)
        elif dtype is float:
            cfg_arg = cfg_obj.getfloat('command_line_arguments',label)
        elif dtype is str:
            cfg_arg = cfg_obj.get('command_line_arguments',label)
        elif dtype is bool:
            cfg_arg = cfg_obj.getboolean('command_line_arguments',label)
        else:
            raise ValueError('dtype must be either int, float, str, or bool')
    except cp.Error:          
        if raise_ex: 
            parser.error('{} is a required argument since it was not found in {}'.format(label,cfg_file))
        else:
            return None
    else:
        return cfg_arg

def update_if_none(ARGS,label,cfg_obj,dtype,parser):
    if ARGS[label] is not None:
        return ARGS[label]
    else:
        return config_fetch(label,cfg_obj,ARGS['args_file'],dtype,parser)        

def update_if_false(ARGS,label,cfg_obj,parser):
    if ARGS[label] == False:
        return ARGS[label]
    else:
        rval = config_fetch(label,cfg_obj,ARGS['args_file'],bool,parser,raise_ex=False)       
        if rval is not None:
            return rval
        else:
            return False

def get_args(extra_args):
    ARGS_dict = {'ckpt_dir':[str,'Directory for storing run time data.'],
            'train_loaddir':[str,'Directory containing training data.'],
            'test_loaddir':[str,'Directory containing testing/validation data.'],
            'filter_file':[str,'Wild-character for filtering data files.'],
            'train_savedir':[str,'Directory to save processed training data'],
            'test_savedir':[str,'Directory to save processed testing/validation data'],
            'num_labels':[int,'Number of class labels.'],
            'data_chan':[int,'Number of channels in input image or volume.'],
            'space_dim':[int,'Number of spatial dimensions. Must be either 2 or 3.'],
            'unet_chan':[int,'Number of U-net input channels.'],
            'unet_blocks':[int,'Number of U-net blocks each with two conv layers.'],
            'aenc_chan':[int,'Number of autoencoder channels.'],
            'aenc_depth':[int,'Depth of autoencoder.'],
            'epochs':[int,'Number of epochs.'],
            'batch':[int,'Batch size.'],
            're_pow':[int,'Power (of norm) for absolute reconstruction error.'],
            'train_num':[int,'Number of samples for training. If zero or negative, use all training samples.'],
            'test_num':[int,'Number of samples for testing/validation. If zero or negative, use all training samples.'],
            'lr':[float,'Learning rate.'],
            'smooth_reg':[float,'Regularization enforcing smoothness.'],
            'devr_reg':[float,'Regularization parameter for anti-devouring loss.'],
            'min_freqs':[float,'Minimum probability of occurance of each class'],
            #'mean':[float,'Mean of data for mean subtraction.'],
            'stdev':[float,'Standard deviation of data for normalization.'],
            'size_dim':[int,'Size of each dimension of input volume or image.'],
            'load_epoch':[int,'Model epoch to load. If negative, does not load model.'],
            'distr':[bool,'If used, distribute model among all GPUs in a node.']
            }

    if extra_args is not None:
        ARGS_dict.update(extra_args)

    parser = argparse.ArgumentParser()
    parser.add_argument('--args_file',type=str,required=True,help='File to read and store run-time arguments')
    for key,(dtype,h) in ARGS_dict.items():
        if dtype is not bool:
            parser.add_argument('--{}'.format(key),default=None,type=dtype,help=h)
        else:
            parser.add_argument('--{}'.format(key),action='store_true',help=h)
    ARGS_inp = vars(parser.parse_args())

    config = cp.ConfigParser()
    config.read(ARGS_inp['args_file'])
    ARGS_inp['distr'] = update_if_false(ARGS_inp,'distr',config,parser)
    for key,(dtype,h) in ARGS_dict.items():
        ARGS_inp[key] = update_if_none(ARGS_inp,key,config,dtype,parser)
    
    if not os.path.exists(ARGS_inp['ckpt_dir']):
        os.makedirs(ARGS_inp['ckpt_dir'])
    if not os.path.exists(ARGS_inp['train_savedir']):
        os.makedirs(ARGS_inp['train_savedir'])
    if not os.path.exists(ARGS_inp['test_savedir']):
        os.makedirs(ARGS_inp['test_savedir'])

    config = cp.ConfigParser()
    config['command_line_arguments'] = {}
    for key,val in ARGS_inp.items():
        config['command_line_arguments'][key] = str(val)
    with open(os.path.join(ARGS_inp['args_file']),'w') as cfg:
        config.write(cfg)

    return ARGS_inp

def write_args(ARGS):
    config = cp.ConfigParser()
    config['command_line_arguments'] = {}
    for key,val in ARGS.items():
        config['command_line_arguments'][key] = str(val)
    with open(os.path.join(ARGS['args_file']),'w') as cfg:
        config.write(cfg)
