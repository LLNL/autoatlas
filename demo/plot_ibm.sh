#!/bin/bash
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -G guests                   #account
#BSUB -o log_plot.txt             #stdout
#BSUB -J tbi_plot                    #name of job
#BSUB -q pbatch                   #queue to use

source ~/.bashrc
conda activate pytorch36
#CUDA_LAUNCH_BLOCKING=1 python demo.py
python plot.py --log_dir chan64_32_labels16_devr10.0_minfreqs0.03_smooth0.1 --num_test 10
python plot.py --log_dir 2mm/chan32_16_labels8_devr1.0_minfreqs0.05_smooth0.2 --num_test 10
python plot.py --log_dir 2mm/chan32_16_labels8_devr10.0_minfreqs0.05_smooth0.2 --num_test 10
python plot.py --log_dir chan64_32_labels16_devr1.0_minfreqs0.03_smooth0.1 --num_test 10
python plot.py --log_dir chan64_32_labels16_devr1.0_minfreqs0.03_smooth0.5 --num_test 10
python plot.py --log_dir chan64_32_labels16_devr1.0_minfreqs0.03_smooth1.0 --num_test 10
python plot.py --log_dir chan64_32_labels8_devr1.0_minfreqs0.05_smooth0.1 --num_test 10
python plot.py --log_dir chan64_32_labels8_devr1.0_minfreqs0.05_smooth1.0 --num_test 10
