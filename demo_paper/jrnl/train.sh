#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH --license=lscratche
#SBATCH -o /p/lustre1/mohan3/Data/TBI/2mm/debug/log.txt

source ~/.bashrc
conda activate pytorch36

aatrain --args_file /p/lustre1/mohan3/Data/TBI/2mm/debug/args.cfg --lr 1e-4 --load_epoch 49 --epochs 100


