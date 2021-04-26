#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G ccp                      #account
#BSUB -o rlearn_str.txt       #stdout
#BSUB -J rlearn_str               #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
module load cuda/11.0.2
conda activate powerai

aarlearn --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/fMRI_flat_images/aatest/aastrun_all_cli.cfg
aarlearn --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/fMRI_flat_images/aatest/aastrage_all_cli.cfg
aarlearn --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/fMRI_flat_images/aatest/aastrun_emb_cli.cfg
aarlearn --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/fMRI_flat_images/aatest/aastrage_emb_cli.cfg


