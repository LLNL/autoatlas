#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH --license=lscratche
#SBATCH -o /p/lustre1/mohan3/Data/TBI/2mm/debug/log_compare.txt

source ~/.bashrc
conda activate pytorch36

aacompare --args_file /p/lustre1/mohan3/Data/TBI/2mm/debug/args.cfg --atlas_dir /p/lustre1/hcpdata/processed/5TT/seg_flip/2mm 


