#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
#SBATCH -D /g/g90/mohan3/Devs/TBI/atlas2d
##SBATCH --license=lscratche
#SBATCH -o /g/g90/mohan3/Devs/TBI/atlas2d/log_plot.txt

source ~/.bashrc
conda activate pytorch36
python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/atlas2d-gray/labels16_unet64_11_aenc8_10_smooth0.01_devr1.0_freqs0.05 --num_test 10 
python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/atlas2d/labels16_unet64_11_aenc8_10_smooth0.01_devr1.0_freqs0.05 --num_test 10
python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/atlas2d/labels16_unet64_11_aenc8_12_smooth0.01_devr1.0_freqs0.05 --num_test 10
python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/atlas2d/labels16_unet32_13_aenc16_12_smooth0.01_devr1.0_freqs0.05 --num_test 10
python plot.py --log_dir /p/lustre1/mohan3/Data/TBI/atlas2d/labels16_unet32_11_aenc16_10_smooth0.1_devr1.0_freqs0.05 --num_test 10
