#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G ccp                      #account
#BSUB -o /p/gpfs1/mohan3/Data/TBI/HCP/2mm/znvol_labs16_smooth0_1_devrr1_0_roir0/infer_log.txt #stdout
#BSUB -J znvol_labs16_smooth0_1_devrr1_0_roir0_infer #name of job
#BSUB -q pbatch                   #queue to use


source ~/.bashrc
module load cuda/11.0.182
conda activate powerai

aainfer --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/2mm/znvol_labs16_smooth0_1_devrr1_0_roir0/infer_cli.cfg

