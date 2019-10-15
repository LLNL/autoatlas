#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -D /usr/WS2/tbidata/workspace_aditya/Devs/autoseg
##SBATCH --license=lscratche
#SBATCH -o /usr/WS2/tbidata/workspace_aditya/Devs/autoseg/checkpoints/log.txt

source ~/.bashrc
conda activate pytorch36
python demo.py
