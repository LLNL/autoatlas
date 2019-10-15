#!/bin/bash
#BSUB -n 1                   #number of nodes
#BSUB -W 1440                      #walltime in minutes
#BSUB -G guests                   #account
#BSUB -o log_ray.txt             #stdout
#BSUB -J tbi_ray                    #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
conda activate pytorch36
python demo.py
