#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G ccp                      #account
#BSUB -o fsgen_log.txt #stdout
#BSUB -J fsgen                    #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
module load cuda/11.0.182
conda activate powerai

aarlearn --cli_args fsgen_cli.cfg


