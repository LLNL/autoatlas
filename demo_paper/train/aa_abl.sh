#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_labs24_smooth0_freqs0_0417/train_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_labs24_devr0_freqs0_0417/train_log.txt
#SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_labs24_rel0_freqs0_0417/train_log.txt

source ~/.bashrc
conda activate pytorch36

#aatrain --cli_args aa_labs24_smooth0_freqs0_0417.cfg
#aatrain --cli_args aa_labs24_devr0_freqs0_0417.cfg
aatrain --cli_args aa_labs24_rel0_freqs0_0417.cfg


