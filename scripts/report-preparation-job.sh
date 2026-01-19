#!/usr/bin/zsh

### MPI tasks
#SBATCH --ntasks=8              # Ask for 8 MPI tasks

### Job name
#SBATCH --job-name=Report-Preparation

### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=output_report_preprocessing_%J.txt

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=01:00:00

### Request a host with a GPU
### If you need two GPUs, change the number accordingly
#SBATCH --gres=gpu:1

### Project id
#SBATCH --account=p0025751

### Partition
#SBATCH --partition=c23g

export CONDA_ENV_NAME=deepseekenv
export CONDA_ROOT=$HOME/miniforge3
export PATH="$CONDA_ROOT/bin:$PATH"

export PROJECT_ROOT="$HOME/jupyterlab/RadVLM"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"  # Add the project root to PYTHONPATH

source $HOME/.bashrc
conda activate $CONDA_ENV_NAME

python $PROJECT_ROOT/src/radvlm/utils/preprocess_reports.py
