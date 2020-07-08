#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G ccp                      #account
#BSUB -o fsreg_log.txt #stdout
#BSUB -J fsreg                    #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
module load cuda/11.0.182
conda activate powerai

aarlearn --cli_args fsstrun_cli.cfg
aarlearn --cli_args fsstrage_cli.cfg
aarlearn --cli_args fsendun_cli.cfg
aarlearn --cli_args fsendage_cli.cfg
aarlearn --cli_args fsdexun_cli.cfg
aarlearn --cli_args fsdexage_cli.cfg
aarlearn --cli_args fsgait_cli.cfg

