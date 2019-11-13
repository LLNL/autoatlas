#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH --license=lscratche
#SBATCH -o logs/pred_aa2.txt

source ~/.bashrc
conda activate pytorch36

#python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt lin
#python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt boost
#python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt nneigh
#python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt mlp

python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc2_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa2 --opt lin
python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc2_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa2 --opt boost
python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc2_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa2 --opt nneigh
python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc2_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa2 --opt mlp

#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa4 --opt lin
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa4 --opt boost
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa4 --opt nneigh
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa4 --opt mlp

#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa8 --opt lin
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa8 --opt boost
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa8 --opt nneigh
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa8 --opt mlp

#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa16 --opt lin
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa16 --opt boost
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa16 --opt nneigh
#python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc16_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa16 --opt mlp

