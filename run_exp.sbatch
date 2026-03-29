#!/bin/bash
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

# Load necessary modules
module purge
module load python/intel/3.8.6

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
if ! python3 - <<'PY'
import importlib
for name in ("numpy", "matplotlib"):
    importlib.import_module(name)
print("deps ok")
PY
then
  python3 -m pip install --user --upgrade pip
  python3 -m pip install --user numpy matplotlib
fi

# Default to all allocated CPUs for process pool.
WORKERS=${SLURM_CPUS_PER_TASK:-1}

# Output root (override by exporting RESULTS_ROOT in your sbatch submission)
RESULTS_ROOT=${RESULTS_ROOT:-results_slurm}

# Run the experiment
srun python3 navigable_min_degree_experiment.py \
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
