#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G guests                   #account
#BSUB -o /p/gscratchr/mohan3/Devs/TBI/HCP_1mm/log.txt #stdout
#BSUB -J tbi                      #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
conda activate pytorch36
python ../src/train.py --space_dim 3 --num_labels 8 --unet_chan 16 --unet_blocks 11 --aenc_chan 4 --aenc_depth 11 --epochs 100 --batch 1 --lr 1e-4 --smooth_reg 0.1 --devr_reg 1.0 --min_freqs 0.05 --train_folder /p/gscratchr/mohan3/Data/T1_decimate/1mm/train --test_folder /p/gscratchr/mohan3/Data/T1_decimate/1mm/test --size_dim 192 --re_pow 2 --distr --log_dir /p/gscratchr/mohan3/Devs/TBI/HCP_1mm --test_num 10

