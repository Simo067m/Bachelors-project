#!/bin/bash
# Ask for hpc queue
#BSUB -q hpc
# Name the job
#BSUB -J pre_train_resnet
# Ask for memory
#BSUB -R "rusage[mem=4GB]"


# Load python module
module load python3/3.11.7


# Activate virtual environment
source bachelor-venv/bin/activate

# Run the script
python3 src/main.py -ptb-xl -bioclinicalbert -resnet18
