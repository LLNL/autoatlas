#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G ccp                      #account
#BSUB -o /p/gpfs1/mohan3/Data/TBI/HCP/2mm/drlearn_wdcy0_095/log.txt #stdout
#BSUB -J drlearn_wdcy0_095         #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
module load cuda/11.0.182
conda activate powerai

drtrain --cli_args train_wdcy0_095.cfg
drinfer --cli_args infer_wdcy0_095.cfg

