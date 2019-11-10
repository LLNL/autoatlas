import numpy as np
import h5py
import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir',type=str,default='./checkpoints/',help='Directory for storing run time data')
ARGS = parser.parse_args()

train_folder = os.path.join(ARGS.log_dir,'train_aa')
test_folder = os.path.join(ARGS.log_dir,'test_aa')
            
neighs = [[0,0,1],  [0,1,-1],[0,1,0], [0,1,1],
          [1,-1,-1],[1,-1,0],[1,-1,1],
          [1,0,-1], [1,0,0], [1,0,1],
          [1,1,-1], [1,1,0], [1,1,1]]

def get_IO(aa_folder,mode):
    aa_files = [os.path.join(aa_folder,f) for f in os.listdir(aa_folder) if f[-18:]=='_T1w_brain_2_aa.h5']
    print('{} files located in {}'.format(len(aa_files),aa_folder))
    aa_ids = [filen.split('/')[-1].split('_')[0] for filen in aa_files]

    neigh_sims,seg_probs,emb_codes = [],[],[]
    for i in range(len(aa_ids)):
        print('Preparing inputs for ID {}'.format(aa_ids[i]))
        with h5py.File(aa_files[i],'r') as fid:
            seg = np.array(fid['segmentation'])   
            code = np.array(fid['embedding'])
            mask = np.array(fid['mask'])[np.newaxis]

        nums,dens = [],[]
        for (Nz,Ny,Nx) in neighs:
            H = np.concatenate((seg[:,-Nz:],seg[:,:-Nz]),axis=1)
            H = np.concatenate((H[:,:,-Ny:],  H[:,:,:-Ny]),   axis=2)
            H = np.concatenate((H[:,:,:,-Nx:],H[:,:,:,:-Nx]), axis=3)
            H = seg*H
            W = np.concatenate((mask[:,-Nz:],  mask[:,:-Nz]), axis=1)
            W = np.concatenate((W[:,:,-Ny:],  W[:,:,:-Ny]),   axis=2)
            W = np.concatenate((W[:,:,:,-Nx:],W[:,:,:,:-Nx]), axis=3)
            W = mask*W
            nums.append(np.sum(H*W,axis=(1,2,3)))
            dens.append(np.sum(W,axis=(1,2,3)))
        neigh_sims.append(np.sum(np.stack(nums,axis=-1),axis=-1)/np.sum(np.stack(dens,axis=-1),axis=-1)) 
        seg_probs.append(np.sum(seg*mask,axis=(1,2,3))/np.sum(mask,axis=(1,2,3)))
        emb_codes.append(code)

    neigh_sims = np.stack(neigh_sims,axis=0)
    seg_probs = np.stack(seg_probs,axis=0)
    emb_codes = np.stack(emb_codes,axis=0)
    np.savez(os.path.join(aa_folder,'{}_inf_inps.npz'.format(mode)),neigh_sims=neigh_sims,seg_probs=seg_probs,emb_codes=emb_codes,ids=np.array(aa_ids))

get_IO(train_folder,'train')
get_IO(test_folder,'test')
