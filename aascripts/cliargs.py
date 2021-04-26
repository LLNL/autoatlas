import argparse
import configparser as cp
import os

HELP_MSG_DICT = {'cli_args':'File containing all command line input arguments.',
        'target_rlearn':'Name of parameter to be predicted.',
        'task_rlearn':'Choose between regression or classification.',
        'train_in_rlearn':'Filename of input features for training.',
        'train_out_rlearn':'Filename of output targets for training.',
        'train_list':'File containing list of training samples.',
        'train_pred_rlearn':'File to store predicted values from train.',
        'train_summ_rlearn':'File to store ML performance metrics from train.',
        'test_in_rlearn':'Filename of input features for testing.',
        'test_out_rlearn':'Filename of output targets for testing.',
        'test_list':'File containing list of testing samples.',
        'test_pred_rlearn':'File to store predicted values from test.',
        'test_summ_rlearn':'File to store ML performance metrics from test.',
        'train_loaddir':'Folder containing training data.',
        'train_savedir':'Folder where output data from training dataset is stored.',
        'test_loaddir':'Folder containing testing data.',
        'test_savedir':'Folder where output data from testing dataset is stored.',
        'train_atlasdir':'Directory where atlas volumes from training dataset are stored. Voxel values of the atlas must be of integer type.',
        'test_atlasdir':'Directory where atlas volumes from testing dataset are stored. Voxel values of the atlas must be of integer type.',
        'ckpt_dir':'Directory for storing run time data.',
        'in_files':'List of input files.',
        'out_files':'List of output files',
        'atlas_files':'List of atlas files.',
        'num_labels':'Number of class labels.',
        'in_dims':'Dimensions separated by comma.',
        'in_chan':'Number of channels in input image or volume.',
        'in_dim':'Number of spatial dimensions. Must be either 2 or 3.',
        'unet_chan':'Number of U-net input channels.',
        'unet_blocks':'Number of U-net blocks each with two conv layers.',
        'aenc_chan':'Number of autoencoder channels.',
        'aenc_depth':'Depth of autoencoder.',
        'epochs':'Number of epochs.',
        'batch':'Batch size.',
        're_pow':'Power (of norm) for absolute reconstruction error.',
        'lr':'Learning rate.',
        'smooth_reg':'Regularization enforcing smoothness.',
        'devr_reg':'Regularization parameter for anti-devouring loss.',
        'min_freqs':'Minimum probability of occurance of each class',
        'load_epoch':'Model epoch to load. If negative, does not load model.',
        'distr':'If used, distribute model among all GPUs in a node.',
        'rl_tag':'Predict for chosen tag',
        'rl_type':'Must be either regression or classification.',
        }

def config_fetch(label,cfg_obj,cfg_file,dtype,parser,raise_ex=True):   
    try:
        if dtype is int:
            cfg_arg = cfg_obj.getint('command_line_input_arguments',label)
        elif dtype is float:
            cfg_arg = cfg_obj.getfloat('command_line_input_arguments',label)
        elif dtype is str:
            cfg_arg = cfg_obj.get('command_line_input_arguments',label)
        elif dtype is bool:
            cfg_arg = cfg_obj.getboolean('command_line_input_arguments',label)
        else:
            raise ValueError('dtype must be either int, float, str, or bool')
    except cp.Error:          
        if raise_ex: 
            parser.error('{} is a required argument and was not found in {}'.format(label,cfg_file))
        else:
            return None
    else:
        return cfg_arg

def update_if_none(ARGS,label,cfg_obj,dtype,parser):
    if ARGS[label] is not None:
        return ARGS[label]
    else:
        return config_fetch(label,cfg_obj,ARGS['cli_args'],dtype,parser)        

def update_if_false(ARGS,label,cfg_obj,parser):
    if ARGS[label] == True:
        return ARGS[label]
    else:
        rval = config_fetch(label,cfg_obj,ARGS['cli_args'],bool,parser,raise_ex=False)       
        if rval is not None:
            return rval
        else:
            return False

def get_parser(ARGS_inp, ret_dict):
    ARGS_dict = {'cli_args':[str,'File containing all command line input arguments.'],
                 'cli_save':[str,'File to save CLI arguments.']}
    if ARGS_inp is not None:
        ARGS_dict.update(ARGS_inp)

    parser = argparse.ArgumentParser()
    for key,(dtype,h) in ARGS_dict.items():
        if dtype is not bool:
            parser.add_argument('--{}'.format(key),default=None,type=dtype,help=h)
        else:
            parser.add_argument('--{}'.format(key),action='store_true',help=h)
    if ret_dict:
        return parser, ARGS_dict
    else:
        return parser

def get_args(parser, ARGS_labs):
    ARGS_inp = vars(parser.parse_args())
    
    config = cp.ConfigParser()
    config.read(ARGS_inp['cli_args'])
    for key,(dtype,h) in ARGS_labs.items():
        if dtype is bool: 
            ARGS_inp[key] = update_if_false(ARGS_inp,key,config,parser)
        else:
            ARGS_inp[key] = update_if_none(ARGS_inp,key,config,dtype,parser)
    
    #if not os.path.exists(ARGS_inp['train_savedir']):
    #    os.makedirs(ARGS_inp['train_savedir'])
    #if not os.path.exists(ARGS_inp['test_savedir']):
    #    os.makedirs(ARGS_inp['test_savedir'])

    return ARGS_inp

def write_args(ARGS,out_file):
    config = cp.ConfigParser()
    config['command_line_input_arguments'] = {}
    for key,val in ARGS.items():
        config['command_line_input_arguments'][key] = str(val)
    with open(out_file,'w') as cfg:
        config.write(cfg)
