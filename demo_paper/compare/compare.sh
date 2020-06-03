#!/bin/bash
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_labs16_smooth0_1_devrm0_8_roim1_2_lb3/compare_log.txt

source ~/.bashrc
conda activate pytorch36

aacompare --cli_args aa_labs16_smooth0_1_devrm0_8_roim1_2_lb3.cfg

