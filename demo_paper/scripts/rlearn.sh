#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH --license=lscratche
#SBATCH -o /p/lustre1/mohan3/Data/TBI/2mm/debug/log_process.txt

source ~/.bashrc
conda activate pytorch36

aarlearn --args_file /p/lustre1/mohan3/Data/TBI/2mm/debug/args.cfg --tag Strength_Unadj --type regression --gtfile ../data/unrestricted_kaplan7_4_1_2019_18_31_31.csv 

