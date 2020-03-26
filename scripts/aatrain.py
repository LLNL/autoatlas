from autoatlas import AutoAtlas
from autoatlas.data import NibData
import os
from .cliargs import get_args,write_args

def main():
    ARGS = get_args(None)

    #Datasets
    train_files = [os.path.join(ARGS['train_loaddir'],f) for f in os.listdir(ARGS['train_loaddir']) if ARGS['filter_file'] in f]
    if ARGS['train_num'] > 0:
        train_files = train_files[:ARGS['train_num']]
    dims = [ARGS['size_dim'] for _ in range(ARGS['space_dim'])]
    train_data = NibData(train_files,dims,None,ARGS['stdev'])
    if ARGS['test_loaddir'] is not None:
        test_files = [os.path.join(ARGS['test_loaddir'],f) for f in os.listdir(ARGS['test_loaddir']) if ARGS['filter_file'] in f]
        if ARGS['test_num']>0:
            test_files = test_files[:ARGS['test_num']]
        valid_data = NibData(test_files,dims,None,ARGS['stdev'])

    #NN Model
    if ARGS['load_epoch'] >= 0:
        autoseg = AutoAtlas(ARGS['num_labels'],sizes=dims,data_chan=ARGS['data_chan'],smooth_reg=ARGS['smooth_reg'],devr_reg=ARGS['devr_reg'],min_freqs=ARGS['min_freqs'],batch=ARGS['batch'],lr=ARGS['lr'],unet_chan=ARGS['unet_chan'],unet_blocks=ARGS['unet_blocks'],aenc_chan=ARGS['aenc_chan'],aenc_depth=ARGS['aenc_depth'],re_pow=ARGS['re_pow'],distr=ARGS['distr'],device='cuda',checkpoint_dir=ARGS['ckpt_dir'],load_checkpoint_epoch=ARGS['load_epoch'])
    elif ARGS['load_epoch'] == -1:
        autoseg = AutoAtlas(ARGS['num_labels'],sizes=dims,data_chan=ARGS['data_chan'],smooth_reg=ARGS['smooth_reg'],devr_reg=ARGS['devr_reg'],min_freqs=ARGS['min_freqs'],batch=ARGS['batch'],lr=ARGS['lr'],unet_chan=ARGS['unet_chan'],unet_blocks=ARGS['unet_blocks'],aenc_chan=ARGS['aenc_chan'],aenc_depth=ARGS['aenc_depth'],re_pow=ARGS['re_pow'],distr=ARGS['distr'],device='cuda',checkpoint_dir=ARGS['ckpt_dir'])
    else:
        raise ValueError('load_epoch must be greater than or equal to -1')

    #Training
    prev_epoch = ARGS['load_epoch']
    for epoch in range(prev_epoch+1,ARGS['epochs']):
        print("Epoch {}".format(epoch))
        autoseg.train(train_data)
        if ARGS['test_loaddir'] is not None:
            autoseg.test(valid_data)
        autoseg.checkpoint(epoch)
        ARGS['load_epoch'] = epoch
        write_args(ARGS)

#End of function main()

if __name__ == "__main__": 
    main() 
