#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH --license=lscratche
#SBATCH -o /p/lustre1/mohan3/Data/TBI/2mm/debug/log.txt

source ~/.bashrc
conda activate pytorch36

aaprocess --args_file /p/lustre1/mohan3/Data/TBI/2mm/debug/args.cfg --load_epoch 99 --train_num 10 --test_num 10


