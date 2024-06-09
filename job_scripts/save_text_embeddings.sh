#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J Save_text_embeddings
# -- choose queue --
#BSUB -q hpc
# -- specify that we need 4GB of memory per core/slot --
# so when asking for 4 cores, we are really asking for 4*4GB=16GB of memory 
# for this job. 
#BSUB -R "rusage[mem=4GB]"
# -- Output File --
#BSUB -o save_text_embeddings.out
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 02:00 
# -- Number of cores requested -- 
#BSUB -n 4
# -- Specify the distribution of the cores: on a single node --
#BSUB -R "span[hosts=1]"
# -- end of LSF options -

# Load python module
module load python3/3.11.7

# Load cuda
module load cuda/12.1

# Activate virtual environment
source bachelor-venv/bin/activate

python3 -u src/save_text_embeddings.py