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

# Launch training with Determined's distributed launcher
# This handles all the distributed setup automatically
echo "=== Launching Script ==="
python $PROJECT_ROOT/tests/check-dataset-similarity-with-sample-fraction.py

echo ""
echo "=== Search Complete ==="
