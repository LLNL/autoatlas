#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G ccp                      #account
#BSUB -o aareg1_log_emb4.txt       #stdout
#BSUB -J aareg1_emb4               #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
module load cuda/11.0.182
conda activate powerai

aarlearn --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_005_devrr0_1_devrm0_9/aastrun_emb_cli.cfg
aarlearn --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_005_devrr0_1_devrm0_9/aastrage_emb_cli.cfg
aarlearn --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_005_devrr0_1_devrm0_9/aaendun_emb_cli.cfg
aarlearn --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_005_devrr0_1_devrm0_9/aaendage_emb_cli.cfg

