#!/bin/bash
# Ask for a queue with gpu
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
# Name the job
#BSUB -J Train_linear_classifier_20_pre
# Ask for memory
#BSUB -R "rusage[mem=2GB]"
# Add walltime
#BSUB -W 04:00
# Add number of cores
#BSUB -n 4
# Specify number of hosts
#BSUB -R "span[hosts=1]"
# Name output file
#BSUB -o Linear_Classifier_20_epochs_pre%J.out
# Get an email when execution ends
#BSUB -N

# Load python module
module load python3/3.11.7

# Load cuda
module load cuda/12.1

# Activate virtual environment
source bachelor-venv/bin/activate

# Run the script
python3 -u src/main.py -pre-split -ptb-xl -bioclinicalbert -resnet18 -log-wandb -wandb-project Bachelors-project -run-config task=train_linear_classifier epochs=50 save_name=Linear_classifier_Resnet18_20_epochs pre_trained_ecg_model=Resnet18_pre_trained_20_epochs batch-size=32