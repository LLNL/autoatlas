#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -o log_znrt.txt 

source ~/.bashrc
conda activate pytorch36

python hcp_prep.py
python tbi_prep.py
