#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -o /g/g90/mohan3/Devs/TBI/autoatlas/log_encode_256_1.txt

source ~/.bashrc
conda activate pytorch36
#python encode.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05 --load_epoch 399
#python encode.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --load_epoch 399
#python encode.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --load_epoch 399
python encode.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc256_11_labels1_smooth0.1_devr1.0_freqs0.05 --load_epoch 399

