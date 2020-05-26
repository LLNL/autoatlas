#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -o /p/lustre1/mohan3/Data/TBI/HCP/2mm/abl_devr/infer_log.txt

source ~/.bashrc
conda activate pytorch36

aainfer --cli_args abl_devr.cfg --load_epoch 249


