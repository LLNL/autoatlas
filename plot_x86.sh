#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -o /g/g90/mohan3/Devs/TBI/autoatlas/log_plot.txt

source ~/.bashrc
conda activate pytorch36
python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05 --num_test 10
python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/norm1_linbott_aenc16_11_labels16_smooth0.01_devr0.1_freqs0.05 --num_test 10

#python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/norm1_aenc16_10_labels16_smooth0.1_devr1.0_freqs0.05 --num_test 10
#python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/norm1_aenc16_10_labels16_smooth0.01_devr0.1_freqs0.05 --num_test 10
#python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/norm1_aenc1_10_labels16_smooth0.01_devr0.1_freqs0.05 --num_test 10
  
#python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/aenc_16_8_labels16_smooth0.1_devr1.0_freqs0.05 --num_test 10
#python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/aenc_16_8_labels16_smooth0.2_devr1.0_freqs0.05 --num_test 10
#python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/labels16_smooth0.1_devr1.0_freqs0.05 --num_test 10
#python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/labels8_smooth0.2_devr1.0_freqs0.1 --num_test 10
#python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/labels8_smooth0.1_devr1.0_freqs0.1 --num_test 10
#python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/labels8_smooth0.5_devr1.0_freqs0.1 --num_test 10
