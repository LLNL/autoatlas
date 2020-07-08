#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G ccp                      #account
#BSUB -o /p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_005_devrr0_1_devrm0_9_emb16/aastrun_log.txt #stdout
#BSUB -J mxvol_labs16_smooth0_005_devrr0_1_devrm0_9_emb16_aastrun #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
module load cuda/11.0.182
conda activate powerai

aarlearn --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_005_devrr0_1_devrm0_9_emb16/aastrun_cli.cfg


