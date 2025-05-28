#!/usr/bin/zsh

### Job name
#SBATCH --job-name=DS-VL2-Train

### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=output_%J.txt

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=80:00:00

### Request a host with a GPU
### If you need two GPUs, change the number accordingly
#SBATCH --gres=gpu:2

### if needed: switch to your working directory (where you saved your program)
# if you installed Miniforge to a different location, change the path accordingly


module load GCCcore/.9.3.0
module load Python/3.9.6
module load cuDNN/8.1.1.33-CUDA-11.2.1

export CONDA_ENV_NAME=deepseekenv
export CONDA_ROOT=$HOME/miniforge3
export PATH="$CONDA_ROOT/bin:$PATH"

export PROJECT_ROOT="$HOME/RadVLM"

conda activate $CONDA_ENV_NAME

cd $PROJECT_ROOT/src/radvlm

python train_deepseek.py

