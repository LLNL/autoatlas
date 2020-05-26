#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/wspace/fs_rlearn_log.txt

source ~/.bashrc
conda activate pytorch36

aarlearn --cli_args fs_args.cfg


