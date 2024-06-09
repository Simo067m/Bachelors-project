#!/bin/bash
# Ask for a queue with gpu
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
# Name the job
#BSUB -J ResNet18_clip_20_epochs_prompt
# Ask for memory
#BSUB -R "rusage[mem=4GB]"
# Add walltime
#BSUB -W 12:00
# Add number of cores
#BSUB -n 4
# Specify number of hosts
#BSUB -R "span[hosts=1]"
# Name output file
#BSUB -o ResNet18_clip_20_epochs_prompt.out

# Load python module
module load python3/3.11.7

# Load cuda
module load cuda/12.1

# Activate virtual environment
source bachelor-venv/bin/activate

# Run the script
python3 -u src/main.py -pre-split -ptb-xl -bioclinicalbert -resnet18-bottleneck -log-wandb -use_standard_text_prompt -wandb-project Bachelors-project -wandb-name ResNet18_clip_20_epochs_prompt -run-config task=ECG_pre_training epochs=20 save_name=ResNet18_clip_20_epochs_prompt batch-size=128