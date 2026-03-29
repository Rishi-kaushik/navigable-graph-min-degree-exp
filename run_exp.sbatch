#!/bin/bash
#SBATCH --job-name=navigable_graph_sparsity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --output=navigable_graph_sparsity_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rk4870@nyu.edu

# Load necessary modules
module purge
module load python/intel/3.8.6
module load openmpi/intel/4.0.5

# Install required packages to user directory
# Core dependencies: numpy for computations, mpi4py for future parallelization
# typing_extensions for older Python compatibility (if needed)
python3 -m pip install --user numpy mpi4py

# Run the sparsity analysis with mpirun for future compatibility
mpirun python3 /home/rk4870/set_cover_navigable/scripts/analyze_graphs.py
