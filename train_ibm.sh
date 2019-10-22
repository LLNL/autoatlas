#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G guests                   #account
#BSUB -o 2mm/chan32_16_labels8_devr10.0_minfreqs0.1_smooth0.1/log.txt             #stdout
#BSUB -J tbi                    #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
conda activate pytorch36
#python train.py --num_labels 16 --unet_chan 64 --unet_blocks 9 --aenc_chan 32 --aenc_depth 8 --epochs 300 --batch 4 --num_test 100 --lr 1e-5 --smooth_reg 0.1 --devr_reg 10.0 --min_freqs 0.03 --train_folder /p/gscratchr/mohan3/Data/T1_decimate/3mm/train --test_folder /p/gscratchr/mohan3/Data/T1_decimate/3mm/test --dim 64 --log_dir chan64_32_labels16_devr10.0_minfreqs0.03_smooth0.1 --load_epoch 117

python train.py --num_labels 8 --unet_chan 32 --unet_blocks 11 --aenc_chan 16 --aenc_depth 10 --epochs 200 --batch 4 --num_test 100 --lr 1e-4 --smooth_reg 0.1 --devr_reg 10.0 --min_freqs 0.1 --train_folder /p/gscratchr/mohan3/Data/T1_decimate/2mm/train --test_folder /p/gscratchr/mohan3/Data/T1_decimate/2mm/test --dim 96 --log_dir 2mm/chan32_16_labels8_devr10.0_minfreqs0.1_smooth0.1
