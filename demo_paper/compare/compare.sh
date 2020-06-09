#!/bin/bash
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/aa_labs16_smooth0_1_devrr1_0_devrm0_8_uchan32_scroi/compare_log.txt

source ~/.bashrc
conda activate pytorch36

aacompare --cli_args aa_labs16_smooth0_1_devrr1_0_devrm0_8_uchan32_scroi.cfg

