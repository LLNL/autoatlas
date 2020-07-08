#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/drlearn_wdcy0_01/train_log.txt

source ~/.bashrc
conda activate pytorch36

drtrain --cli_args train_wdcy0_01.cfg

