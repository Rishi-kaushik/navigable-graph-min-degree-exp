#!/usr/bin/env bash
#SBATCH --job-name=min_degree
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --output=/scratch/%u/min_degree_%j.out
#SBATCH --error=/scratch/%u/min_degree_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rk4870@nyu.edu

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/rk4870/navigable-graph-min-degree-exp}
RESULTS_ROOT=${RESULTS_ROOT:-/scratch/$USER/min_degree_results}
WORKERS=${SLURM_CPUS_PER_TASK:-1}

# Use Anaconda module on Torch (includes pip + common scientific stack)
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load anaconda3/2025.06
fi

# Keep installs out of $HOME
export PYTHONUSERBASE=${PYTHONUSERBASE:-/scratch/$USER/python-user-base}
export PATH="$PYTHONUSERBASE/bin:$PATH"

# Prevent BLAS oversubscription when using process pool workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$RESULTS_ROOT" "$PYTHONUSERBASE"
cd "$REPO_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

echo "python: $(command -v $PYTHON_BIN || true)"
"$PYTHON_BIN" -V

# Install deps only if missing (Anaconda often already includes them)
if ! "$PYTHON_BIN" - <<'PY'
import numpy
import matplotlib
print("deps ok")
PY
then
  "$PYTHON_BIN" -m pip install --user --upgrade pip
  "$PYTHON_BIN" -m pip install --user numpy matplotlib
fi

"$PYTHON_BIN" navigable_min_degree_experiment.py \
  --distribution gaussian \
  --normalize-points \
  --seed 0 \
  --points-start 50 --points-end 6400 --points-multiplier 2 \
  --dims-start 5 --dims-end 50 --dims-step 5 \
  --experiments-per-cell 5 \
  --workers "$WORKERS" \
  --results-root "$RESULTS_ROOT" \
  --csv-name min_degree_heatmap.csv \
  --avg-heatmap-name avg_min_degree_heatmap.png \
  --max-heatmap-name max_min_degree_heatmap.png
