#!/bin/bash
set -e

echo "=== Starting MedGemma DPO Training ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Use the mounted path instead of $HOME
export CONDA_ROOT="/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/miniconda3"
export CONDA_ENV_NAME=medgemmaenv
export PROJECT_ROOT="/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "CONDA_ROOT: $CONDA_ROOT"
echo "CONDA_ENV_NAME: $CONDA_ENV_NAME"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo ""

# Initialize conda
echo "Activating conda environment..."
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

# Launch training with Determined's distributed launcher
# This handles all the distributed setup automatically
echo "=== Launching Training ==="
python $PROJECT_ROOT/src/radvlm/train_medgemma_dpo.py

echo ""
echo "=== Training Complete ==="