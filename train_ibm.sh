#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G guests                   #account
#BSUB -o labels16_devr1.0_minfreqs0.03_smooth0.1/log.txt             #stdout
#BSUB -J tbi                    #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
conda activate pytorch36
#CUDA_LAUNCH_BLOCKING=1 python demo.py
python demo.py
