#!/usr/bin/env bash
#SBATCH --job-name=min_degree
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --output=min_degree_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rk4870@nyu.edu

set -euo pipefail

# Load a Python module if available (cluster-dependent)
if command -v module >/dev/null 2>&1; then
  module purge || true
  # Try common module names first; keep this list short and generic.
  for m in \
    python \
    python3 \
    python/3.11 \
    python/3.10 \
    python/3.9 \
    python/3.8 \
    anaconda3 \
    miniconda3
  do
    module load "$m" >/dev/null 2>&1 && break
  done
fi

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "error: python not found (check 'module avail python')" >&2
  exit 1
fi

# Repo location on the remote cluster
REPO_DIR=/home/rk4870/navigable-graph-min-degree-exp
cd "$REPO_DIR"

# Avoid oversubscription: each worker process uses 1 BLAS thread.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Install required packages to user directory (only if missing)
if ! "$PYTHON_BIN" - <<'PY'
import importlib
for name in ("numpy", "matplotlib"):
    importlib.import_module(name)
print("deps ok")
PY
then
  "$PYTHON_BIN" -m pip install --user --upgrade pip
  "$PYTHON_BIN" -m pip install --user --no-cache-dir numpy matplotlib
fi

# Default to all allocated CPUs for process pool.
WORKERS=${SLURM_CPUS_PER_TASK:-1}

# Output root (override by exporting RESULTS_ROOT in your sbatch submission)
RESULTS_ROOT=${RESULTS_ROOT:-results_slurm}

# Run the experiment
srun "$PYTHON_BIN" navigable_min_degree_experiment.py \
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
