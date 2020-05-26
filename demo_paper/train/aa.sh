#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi16_rad1_5_roi1/train_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi24_rad1_5_roi1/train_log.txt
#SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi32_rad1_5_roi1/train_log.txt

##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi16/train_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi16_rad1_5/train_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi24/train_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi24_rad1_5/train_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi32/train_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi32_rad2/train_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi32_rad1_5/train_log.txt

source ~/.bashrc
conda activate pytorch36

#aatrain --cli_args aa_roi16_rad1_5_roi1.cfg
#aatrain --cli_args aa_roi24_rad1_5_roi1.cfg
aatrain --cli_args aa_roi32_rad1_5_roi1.cfg

#aatrain --cli_args aa_roi16.cfg --load_epoch 225
#aatrain --cli_args aa_roi16_rad1_5.cfg
#aatrain --cli_args aa_roi24.cfg --load_epoch 164 
#aatrain --cli_args aa_roi24_rad1_5.cfg
#aatrain --cli_args aa_roi32.cfg --load_epoch 82
#aatrain --cli_args aa_roi32_rad2.cfg --load_epoch 75
#aatrain --cli_args aa_roi32_rad1_5.cfg --load_epoch 72


