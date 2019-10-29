from segmenter import AutoSegmenter
from data import HCPDataset,CelebDataset
import os
import argparse
import configparser as cp

parser = argparse.ArgumentParser()
parser.add_argument('--num_labels',type=int,default=4,help='Number of class labels')
parser.add_argument('--data_chan',type=int,default=1,help='Number of channels in input image or volume.')
parser.add_argument('--space_dim',type=int,default=3,help='Number of spatial dimensions. Must be either 2 or 3')
parser.add_argument('--unet_chan',type=int,default=32,help='Number of U-net input channels')
parser.add_argument('--unet_blocks',type=int,default=9,help='Number of U-net blocks each with two conv layers')
parser.add_argument('--aenc_chan',type=int,default=16,help='Number of autoencoder channels')
parser.add_argument('--aenc_depth',type=int,default=8,help='Depth of autoencoder')
parser.add_argument('--epochs',type=int,default=1,help='Number of epochs')
parser.add_argument('--batch',type=int,default=1,help='Batch size')
parser.add_argument('--re_pow',type=int,default=1,help='Power (of norm) for absolute reconstruction error')
parser.add_argument('--num_test',type=int,default=1,help='Number of samples for testing/validation')
parser.add_argument('--lr',type=float,default=1e-4,help='Learning rate')
parser.add_argument('--smooth_reg',type=float,default=1.0,help='Regularization enforcing smoothness')
parser.add_argument('--devr_reg',type=float,default=1.0,help='Regularization parameter for anti-devouring loss')
parser.add_argument('--min_freqs',type=float,default=0.01,help='Minimum probability of occurance of each class')
parser.add_argument('--log_dir',type=str,default='./checkpoints/',help='Directory for storing run time data')
parser.add_argument('--load_epoch',type=int,default=-1,help='Model epoch to load')
parser.add_argument('--train_folder',type=str,default='/p/gscratchr/mohan3/Data/T1_decimate/3mm/train',help='Directory containing training data')
parser.add_argument('--test_folder',type=str,default='/p/gscratchr/mohan3/Data/T1_decimate/3mm/test',help='Directory containing testing/validation data')
#parser.add_argument('--mean',type=float,default=None,help='Mean of data for mean subtraction')
parser.add_argument('--stdev',type=float,default=286.3180,help='Standard deviation of data for normalization')
parser.add_argument('--size_dim',type=int,default=64,help='Size of each dimension of input volume or image')
ARGS = parser.parse_args()
    
if not os.path.exists(ARGS.log_dir):
    os.makedirs(ARGS.log_dir)
config = cp.ConfigParser()
config['command_line_arguments'] = {}
for key,val in vars(ARGS).items():
    config['command_line_arguments'][key] = str(val)
with open(os.path.join(ARGS.log_dir,'args.cfg'),'w') as cfg:
    config.write(cfg)

#Parameters
entr_reg = 0.0

#Datasets
if ARGS.space_dim==3:
    train_files = [os.path.join(ARGS.train_folder,f) for f in os.listdir(ARGS.train_folder) if f[-7:]=='.nii.gz']
    test_files = [os.path.join(ARGS.test_folder,f) for f in os.listdir(ARGS.test_folder) if f[-7:]=='.nii.gz'][:ARGS.num_test]
    dims = [ARGS.size_dim,ARGS.size_dim,ARGS.size_dim]
    train_data = HCPDataset(train_files,dims,None,ARGS.stdev)
    valid_data = HCPDataset(test_files,dims,None,ARGS.stdev)
elif ARGS.space_dim==2:
    dims = [ARGS.size_dim,ARGS.size_dim]
    train_data = CelebDataset(ARGS.train_folder,dims=dims,mean=None,stdev=ARGS.stdev) 
    valid_data = CelebDataset(ARGS.test_folder,num=ARGS.num_test,dims=dims,mean=None,stdev=ARGS.stdev) 
else:
    raise ValueError('Argument space_dim must be either 2 or 3')

#NN Model
if ARGS.load_epoch >= 0:
    autoseg = AutoSegmenter(ARGS.num_labels,dim=ARGS.space_dim,data_chan=ARGS.data_chan,smooth_reg=ARGS.smooth_reg,devr_reg=ARGS.devr_reg,entr_reg=entr_reg,min_freqs=ARGS.min_freqs,batch=ARGS.batch,lr=ARGS.lr,unet_chan=ARGS.unet_chan,unet_blocks=ARGS.unet_blocks,aenc_chan=ARGS.aenc_chan,aenc_depth=ARGS.aenc_depth,re_pow=ARGS.re_pow,device='cuda',checkpoint_dir=ARGS.log_dir,load_checkpoint_epoch=ARGS.load_epoch)
elif ARGS.load_epoch == -1:
    autoseg = AutoSegmenter(ARGS.num_labels,dim=ARGS.space_dim,data_chan=ARGS.data_chan,smooth_reg=ARGS.smooth_reg,devr_reg=ARGS.devr_reg,entr_reg=entr_reg,min_freqs=ARGS.min_freqs,batch=ARGS.batch,lr=ARGS.lr,unet_chan=ARGS.unet_chan,unet_blocks=ARGS.unet_blocks,aenc_chan=ARGS.aenc_chan,aenc_depth=ARGS.aenc_depth,re_pow=ARGS.re_pow,device='cuda',checkpoint_dir=ARGS.log_dir)
else:
    raise ValueError('load_epoch must be greater than or equal to -1')

#Training
for epoch in range(ARGS.load_epoch+1,ARGS.epochs):
    print("Epoch {}".format(epoch))
    autoseg.train(train_data)
    autoseg.test(valid_data)
    autoseg.checkpoint(epoch)
