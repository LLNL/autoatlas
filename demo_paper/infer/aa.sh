#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi16/infer_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi24/infer_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi32/infer_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi32_rad1_5/infer_log.txt
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi32_rad2/infer_log.txt

#SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi16_rad1_5_roi1/infer_log.txt

source ~/.bashrc
conda activate pytorch36

#aainfer --cli_args aa_roi16.cfg
#aainfer --cli_args aa_roi24.cfg
#aainfer --cli_args aa_roi32.cfg
#aainfer --cli_args aa_roi32_rad1_5.cfg
#aainfer --cli_args aa_roi32_rad2.cfg

aainfer --cli_args aa_roi16_rad1_5_roi1.cfg

