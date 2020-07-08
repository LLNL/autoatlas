#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 120                      #walltime in minutes
#BSUB -G ccp                      #account
#BSUB -o /p/gpfs1/mohan3/Data/TBI/HCP/2mm/znvol_labs16_smooth0_1_devrr1_0_roir0/compare_log.txt #stdout
#BSUB -J znvol_labs16_smooth0_1_devrr1_0_roir0_compare #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
module load cuda/11.0.182
conda activate powerai

aacompare --cli_args /p/gpfs1/mohan3/Data/TBI/HCP/2mm/znvol_labs16_smooth0_1_devrr1_0_roir0/compare_cli.cfg 

