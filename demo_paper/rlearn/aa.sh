#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/wspace/aa_all_log.txt

source ~/.bashrc
conda activate pytorch36

aarlearn --cli_args aa_all_args.cfg


