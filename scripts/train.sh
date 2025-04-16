#!/usr/bin/zsh

### Job name
#SBATCH --job-name=MyDugleJob

### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=output_%J.txt

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=80

### Request a host with a GPU
### If you need two GPUs, change the number accordingly
#SBATCH --gres=gpu:1

### if needed: switch to your working directory (where you saved your program)
# if you installed Miniforge to a different location, change the path accordingly

export CONDA_ENV_NAME=dugle
export CONDA_ROOT=$HOME/miniforge3
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate $CONDA_ENV_NAME