#!/bin/bash
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH --license=lscratche
#SBATCH -o logs/pred_aa16.txt

source ~/.bashrc
conda activate pytorch36

#python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt lin
#python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt svm
#python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt nneigh
#python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt mlp

#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa4 --opt lin
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa4 --opt nneigh
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa4 --opt svm
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa4 --opt mlp

#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa8 --opt lin
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa8 --opt svm
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa8 --opt nneigh
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa8 --opt mlp

python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa16 --opt lin
python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa16 --opt svm
python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa16 --opt nneigh
python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa16 --opt mlp

