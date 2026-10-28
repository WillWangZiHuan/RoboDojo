#!/usr/bin/env bash
# Submit from the repository root; choose partition/GPU/time via sbatch flags.
# Usage: sbatch ... experiments/seven_blocks/slurm.sh /path/to/activate.sh smoke
set -euo pipefail
activation=${1:?Pass the existing RoboDojo environment activation script}
mode=${2:?Pass the run mode: smoke, manipulation, or preview}
shift 2
source "$activation"
cd "${SLURM_SUBMIT_DIR:?Run through sbatch from the repository root}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
srun python -u -m experiments.seven_blocks.run --mode "$mode" --headless \
  --output "eval_result/seven_blocks/${mode}_${SLURM_JOB_ID}" "$@"
