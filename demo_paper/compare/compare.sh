#!/bin/bash
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi16/compare_log.txt
#SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_roi24/compare_log.txt

source ~/.bashrc
conda activate pytorch36

#aacompare --cli_args aa_roi16.cfg
aacompare --cli_args aa_roi24.cfg


