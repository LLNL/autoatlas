#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G ccp                      #account
#BSUB -o /p/gpfs1/mohan3/Data/TBI/HCP/2mm/znvol_HCP2TBI_smooth0_2_devr1_0_znrt/train_log.txt #stdout
#BSUB -J znvol_HCP2TBI_smooth0_2_devr1_0_znrt_train_launch4 #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
module load cuda/11.0.2
conda activate powerai

aatrain --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/2mm/znvol_HCP2TBI_smooth0_2_devr1_0_znrt/train_cli.cfg
