from segmenter import AutoSegmenter
from data import HCPDataset
import os
import argparse
import configparser as cp

parser = argparse.ArgumentParser()
parser.add_argument('--num_labels',type=int,default=4,help='Number of class labels')
parser.add_argument('--unet_chan',type=int,default=32,help='Number of U-net input channels')
parser.add_argument('--unet_blocks',type=int,default=9,help='Number of U-net blocks each with two conv layers')
parser.add_argument('--aenc_chan',type=int,default=16,help='Number of autoencoder channels')
parser.add_argument('--aenc_depth',type=int,default=8,help='Depth of autoencoder')
parser.add_argument('--epochs',type=int,default=1,help='Number of epochs')
parser.add_argument('--batch',type=int,default=1,help='Batch size')
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
parser.add_argument('--dim',type=int,default=64,help='Dimension along each of three dimensions for input volumes')
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
dims = [ARGS.dim,ARGS.dim,ARGS.dim]

#Datasets
train_files = [os.path.join(ARGS.train_folder,f) for f in os.listdir(ARGS.train_folder) if f[-7:]=='.nii.gz']
test_files = [os.path.join(ARGS.test_folder,f) for f in os.listdir(ARGS.test_folder) if f[-7:]=='.nii.gz'][:ARGS.num_test]

train_data = HCPDataset(train_files,dims,None,ARGS.stdev)
valid_data = HCPDataset(test_files,dims,None,ARGS.stdev)

#NN Model
if ARGS.load_epoch >= 0:
    autoseg = AutoSegmenter(ARGS.num_labels,smooth_reg=ARGS.smooth_reg,devr_reg=ARGS.devr_reg,entr_reg=entr_reg,min_freqs=ARGS.min_freqs,batch=ARGS.batch,lr=ARGS.lr,unet_chan=ARGS.unet_chan,unet_blocks=ARGS.unet_blocks,aenc_chan=ARGS.aenc_chan,aenc_depth=ARGS.aenc_depth,device='cuda',checkpoint_dir=ARGS.log_dir,load_checkpoint_epoch=ARGS.load_epoch)
elif ARGS.load_epoch == -1:
    autoseg = AutoSegmenter(ARGS.num_labels,smooth_reg=ARGS.smooth_reg,devr_reg=ARGS.devr_reg,entr_reg=entr_reg,min_freqs=ARGS.min_freqs,batch=ARGS.batch,lr=ARGS.lr,unet_chan=ARGS.unet_chan,unet_blocks=ARGS.unet_blocks,aenc_chan=ARGS.aenc_chan,aenc_depth=ARGS.aenc_depth,device='cuda',checkpoint_dir=ARGS.log_dir)
else:
    raise ValueError('load_epoch must be greater than or equal to -1')

#Training
for epoch in range(ARGS.load_epoch+1,ARGS.epochs):
    print("Epoch {}".format(epoch))
    autoseg.train(train_data)
    autoseg.test(valid_data)
    autoseg.checkpoint(epoch)
