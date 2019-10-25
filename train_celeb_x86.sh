#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -D /g/g90/mohan3/Devs/TBI/atlas2d
##SBATCH --license=lscratche
#SBATCH -o /p/lustre1/mohan3/Data/TBI/atlas2d-gray/labels16_unet64_11_aenc8_10_smooth0.01_devr1.0_freqs0.05/log.txt

source ~/.bashrc
conda activate pytorch36
python train.py --num_labels 16 --unet_chan 64 --unet_blocks 11 --aenc_chan 8 --aenc_depth 10 --epochs 400 --batch 256 --num_test 1024 --lr 1e-4 --smooth_reg 0.01 --devr_reg 1.0 --min_freqs 0.05 --train_folder /p/lustre3/mohan3/Data/TBI/CelebAMask-HQ/train --test_folder /p/lustre3/mohan3/Data/TBI/CelebAMask-HQ/test --space_dim 2 --size_dim 64 --stdev 255 --log_dir /p/lustre1/mohan3/Data/TBI/atlas2d-gray/labels16_unet64_11_aenc8_10_smooth0.01_devr1.0_freqs0.05 
