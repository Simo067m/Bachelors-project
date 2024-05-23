#!/bin/bash
# Ask for a queue with gpu
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
# Name the job
#BSUB -J pre_train_resnet18_1_epochs
# Ask for memory
#BSUB -R "rusage[mem=8GB]"
# Add walltime
#BSUB -W 4:00
# Add number of cores
#BSUB -n 4
# Specify number of hosts
#BSUB -R "span[hosts=1]"
# Name output file
#BSUB -o pre_train_resnet_output_%J.out

# Load python module
module load python3/3.11.7

# Load cuda
module load cuda/12.1


# Activate virtual environment
source bachelor-venv/bin/activate

# Run the script
python3 -u src/main.py -pre-split -ptb-xl -bioclinicalbert -resnet18 -log-wandb -wandb-project Bachelors-project -run-config task=ECG_pre_training epochs=1 save_name=Resnet18_pre_trained
