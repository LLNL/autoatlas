#!/bin/bash
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A ccp
##SBATCH --license=lscratche
#SBATCH -o logs/popt4.txt

source ~/.bashrc
conda activate pytorch36

python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa4 --opt lin --tag Strength_Unadj 
python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa4 --opt nneigh --tag Strength_Unadj
python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa4 --opt boost --tag Strength_Unadj
python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa4 --opt mlp --tag Strength_Unadj

python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt lin --tag Strength_Unadj
python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt nneigh --tag Strength_Unadj
python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt boost --tag Strength_Unadj
python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc4_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt mlp --tag Strength_Unadj

python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa8 --opt lin --tag Strength_Unadj 
python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa8 --opt nneigh --tag Strength_Unadj
python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa8 --opt boost --tag Strength_Unadj
python aa_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_aa8 --opt mlp --tag Strength_Unadj

python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt lin --tag Strength_Unadj
python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt nneigh --tag Strength_Unadj
python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt boost --tag Strength_Unadj
python fs_pred.py --log_dir /p/lustre1/mohan3/Data/TBI/2mm/segin_norm2_linbott_aenc8_11_labels16_smooth0.1_devr1.0_freqs0.05 --pred_file pred_fs --opt mlp --tag Strength_Unadj
