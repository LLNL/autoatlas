#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G guests                   #account
#BSUB -o checkpoints_smooth0.01/log.txt             #stdout
#BSUB -J tbi                    #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
conda activate pytorch36
python demo.py
