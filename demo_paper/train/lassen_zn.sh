#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G ccp                      #account
#BSUB -o /p/gpfs1/mohan3/Data/TBI/HCP/2mm/znvol_labs16_smooth0_01_devrr0_1_roir0/train_log.txt #stdout
#BSUB -J znvol_labs16_smooth0_01_devrr0_1_roir0_train_launch4 #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
module load cuda/11.0.182
conda activate powerai

aatrain --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/2mm/znvol_labs16_smooth0_01_devrr0_1_roir0/train_cli.cfg
