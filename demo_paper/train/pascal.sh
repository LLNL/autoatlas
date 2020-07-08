#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_labs16_smooth0_1_devrr1_0_devrm0_8_uchan32_roim1_2_scroi/train_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_labs16_smooth0_1_devrr1_0_devrm0_8_roir10_0_uchan32_scroi/train_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_labs16_smooth0_1_devrr1_0_devrm0_8_uchan32_scroi/train_log.txt
#SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_labs16_smooth0_1_devrr1_0_roir0_1_nm/train_log.txt

source ~/.bashrc
conda activate pytorch36

#aatrain --cli_args aa_labs16_smooth0_1_devrr1_0_devrm0_8_uchan32_roim1_2_scroi.cfg
#aatrain --cli_args aa_labs16_smooth0_1_devrr1_0_devrm0_8_roir10_0_uchan32_scroi.cfg
#aatrain --cli_args aa_labs16_smooth0_1_devrr1_0_devrm0_8_uchan32_scroi.cfg --load_epoch 120
aatrain --cli_args aa_labs16_smooth0_1_devrr1_0_roir0_1_nm.cfg
