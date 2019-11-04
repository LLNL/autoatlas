#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -o /g/g90/mohan3/Devs/TBI/aapred/log_encode.txt

source ~/.bashrc
conda activate pytorch36
python encode.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/norm2_linbott_aenc256_11_labels1_smooth0.2_devr1.0_freqs0.05 --load_epoch 47 
python encode.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/norm2_linbott_aenc16_11_labels16_smooth0.2_devr1.0_freqs0.05 --load_epoch 31
#python encode.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05 --load_epoch 110
