#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G ccp                      #account
#BSUB -o abl_emb_strage_log.txt      #stdout
#BSUB -J abl_emb_strage              #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
module load cuda/11.0.182
conda activate powerai

aarlearn --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_relr0_0_smooth0_005_devrr0_1_devrm0_9_emb16/aastrage_emb_cli.cfg
aarlearn --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_0_devrr0_1_devrm0_9_emb16/aastrage_emb_cli.cfg
aarlearn --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/2mm/mxvol_labs16_smooth0_005_devrr0_0_devrm0_9_emb16/aastrage_emb_cli.cfg

