#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH --license=lscratche
#SBATCH -o /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels32_smooth0.1_devr1.0_freqs0.05/log.txt

source ~/.bashrc
conda activate pytorch36
#python train.py --space_dim 3 --num_labels 16 --unet_chan 16 --unet_blocks 11 --aenc_chan 2 --aenc_depth 11 --epochs 400 --batch 4 --num_test 100 --lr 1e-4 --smooth_reg 0.1 --devr_reg 1.0 --min_freqs 0.05 --train_folder /p/lustre3/mohan3/Data/TBI/T1_decimate/2mm/train --test_folder /p/lustre3/mohan3/Data/TBI/T1_decimate/2mm/test --size_dim 96 --re_pow 2 --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc2_11_labels16_smooth0.1_devr1.0_freqs0.05

python train.py --space_dim 3 --num_labels 32 --unet_chan 32 --unet_blocks 11 --aenc_chan 4 --aenc_depth 11 --epochs 400 --batch 2 --num_test 100 --lr 1e-4 --smooth_reg 0.1 --devr_reg 1.0 --min_freqs 0.05 --train_folder /p/lustre3/mohan3/Data/TBI/T1_decimate/2mm/train --test_folder /p/lustre3/mohan3/Data/TBI/T1_decimate/2mm/test --size_dim 96 --re_pow 2 --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels32_smooth0.1_devr1.0_freqs0.05

