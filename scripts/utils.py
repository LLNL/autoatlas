import configparser as cp

def get_config(filename):
    config = cp.ConfigParser()
    config.read(filename)
    
    #Parameters
    num_labels = config.getint('command_line_arguments','num_labels')
    data_chan = config.getint('command_line_arguments','data_chan')
    space_dim = config.getint('command_line_arguments','space_dim')
    unet_chan = config.getint('command_line_arguments','unet_chan')
    unet_blocks = config.getint('command_line_arguments','unet_blocks')
    aenc_chan = config.getint('command_line_arguments','aenc_chan')
    aenc_depth = config.getint('command_line_arguments','aenc_depth')
    num_epochs = config.getint('command_line_arguments','epochs')
    batch = config.getint('command_line_arguments','batch')
    re_pow = config.getint('command_line_arguments','re_pow')
    lr = config.getfloat('command_line_arguments','lr')
    smooth_reg = config.getfloat('command_line_arguments','smooth_reg')
    devr_reg = config.getfloat('command_line_arguments','devr_reg')
    min_freqs = config.getfloat('command_line_arguments','min_freqs')
    train_folder = config.get('command_line_arguments','train_folder')
    test_folder = config.get('command_line_arguments','test_folder')
    stdev = config.getfloat('command_line_arguments','stdev')
    size_dim = config.getint('command_line_arguments','size_dim')

    return num_labels,data_chan,space_dim,unet_chan,unet_blocks,aenc_chan,aenc_depth,num_epochs,batch,re_pow,lr,smooth_reg,devr_reg,min_freqs,train_folder,test_folder,stdev,size_dim 
