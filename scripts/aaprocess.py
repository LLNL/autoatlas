import argparse
from autoatlas import AutoAtlas,partition_encode
from autoatlas.data import NibData
import os
import numpy as np
import nibabel as nib
from .cliargs import get_args
import csv

def get_meta(filen):
    dptr = nib.load(filen)
    return dptr.get_affine(),dptr.get_header()

def save_nifti(filen,data,affine,header):
    header.set_data_dtype(data.dtype) 
    header.set_slope_inter(None,None)
    dptr = nib.Nifti1Image(data,affine,header)
    dptr.to_filename(filen)

def write_code(filen,code,row_labels,col_labels):
    assert code.shape[0]==len(row_labels)
    assert code.shape[1]==len(col_labels)-1
    with open(filen,mode='w') as csv_file:
        csv_writer = csv.writer(csv_file,delimiter=',')
        csv_writer.writerow(col_labels)
        for i in range(len(row_labels)):
            data = [row_labels[i]]+['{:.6e}'.format(c) for c in code[i]]
            csv_writer.writerow(data)

def process_data(autoseg,data_files,dims,batch,stdev,save_dir,only_features,num_labels,num_codes):
    row_labels = ['label {}'.format(i) for i in range(num_labels)]
    col_labels = ['encoding {}'.format(i) for i in range(num_codes)]
    col_labels = ['region','normalized volume','normalized surface area'] + col_labels 
    for i in range(0,len(data_files),batch):
        fin = data_files[i:i+batch]
        train_data = NibData(fin,dims,None,stdev)
        segs,recs,masks,embeds,fout,inps = autoseg.process(train_data,ret_input=True)
        assert fin==fout
        for j in range(len(fout)):
            aff,head = get_meta(fout[j])
            _,filen = os.path.split(fout[j])
            filen = filen.split('.')[0]
            vol_meas,area_meas = partition_encode(segs[j],masks[j])
            code = np.stack((vol_meas,area_meas),axis=1)
            code = np.concatenate((code,embeds[j]),axis=1) 
            write_code(os.path.join(save_dir,'{}_aacode.csv'.format(filen)),code,row_labels,col_labels) 
            if not only_features:
                arr = np.transpose(segs[j].astype(np.float32,order='C'),axes=(1,2,3,0))
                save_nifti(os.path.join(save_dir,'{}_aaprob.nii.gz'.format(filen)),arr,aff,head)
                arr = np.argmax(arr,axis=-1)
                save_nifti(os.path.join(save_dir,'{}_aaparts.nii.gz'.format(filen)),arr,aff,head)
                arr = np.transpose(recs[j].astype(np.float32,order='C'),axes=(1,2,3,0)) 
                save_nifti(os.path.join(save_dir,'{}_aarecon.nii.gz'.format(filen)),arr,aff,head)
                save_nifti(os.path.join(save_dir,'{}_aamask.nii.gz'.format(filen)),masks[j],aff,head)
                save_nifti(os.path.join(save_dir,'{}_aainput.nii.gz'.format(filen)),inps[j],aff,head)
 
def main():
    extra_args = {'only_features':[bool,'If specified, only saves the embeddings and partition volume features while NIFTI volume files are not saved.']}
    ARGS = get_args(extra_args)
    dims = [ARGS['size_dim'] for _ in range(ARGS['space_dim'])]

    train_files = [os.path.join(ARGS['train_loaddir'],f) for f in os.listdir(ARGS['train_loaddir']) if ARGS['filter_file'] in f]
    if ARGS['train_num'] > 0:
        train_files = train_files[:ARGS['train_num']]

    if ARGS['test_loaddir'] is not None:
        test_files = [os.path.join(ARGS['test_loaddir'],f) for f in os.listdir(ARGS['test_loaddir']) if ARGS['filter_file'] in f]
        if ARGS['test_num'] > 0:
            test_files = test_files[:ARGS['test_num']]

    if ARGS['load_epoch'] >= 0:
        autoseg = AutoAtlas(ARGS['num_labels'],sizes=dims,data_chan=ARGS['data_chan'],smooth_reg=ARGS['smooth_reg'],devr_reg=ARGS['devr_reg'],min_freqs=ARGS['min_freqs'],batch=ARGS['batch'],lr=ARGS['lr'],unet_chan=ARGS['unet_chan'],unet_blocks=ARGS['unet_blocks'],aenc_chan=ARGS['aenc_chan'],aenc_depth=ARGS['aenc_depth'],re_pow=ARGS['re_pow'],distr=ARGS['distr'],device='cuda',checkpoint_dir=ARGS['ckpt_dir'],load_checkpoint_epoch=ARGS['load_epoch'])
    else:
        raise ValueError('load_epoch must be specified')

    process_data(autoseg,train_files,dims,ARGS['batch'],ARGS['stdev'],ARGS['train_savedir'],ARGS['only_features'],ARGS['num_labels'],ARGS['aenc_chan'])
    process_data(autoseg,test_files,dims,ARGS['batch'],ARGS['stdev'],ARGS['test_savedir'],ARGS['only_features'],ARGS['num_labels'],ARGS['aenc_chan'])

if __name__ == "__main__": 
    main() 
