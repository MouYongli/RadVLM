#!/bin/bash
set -e

# Use the mounted path instead of $HOME
export CONDA_ROOT="/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/miniconda3"
export CONDA_ENV_NAME=medgemmaenv
export PROJECT_ROOT="/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Initialize conda
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

python $PROJECT_ROOT/src/radvlm/evaluate_pretraining_sampling_medgemma.py
