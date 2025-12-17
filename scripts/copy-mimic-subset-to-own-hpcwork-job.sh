#!/usr/bin/zsh

### MPI tasks
#SBATCH --ntasks=8

### Job name
#SBATCH --job-name=Data-Preparation

### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output=output_data_preprocessing_%J.txt

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=02:00:00

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
echo "Project root is: $PROJECT_ROOT"
echo "Home is: $HOME"
echo "HPCWORK is: $HPCWORK"
echo "Conda env name is: $CONDA_ENV_NAME"
# cd $PROJECT_ROOT/src/radvlm/utils
# Read directory names config.py

python $PROJECT_ROOT/src/radvlm/utils/copy_mimic_subset_to_hpcwork.py