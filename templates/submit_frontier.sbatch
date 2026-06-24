#!/bin/bash
#SBATCH -A <YOUR_PROJECT_ID>          # OLCF project/allocation (e.g. MAT123)
#SBATCH -J flibe_npt
#SBATCH -t 02:00:00                   # walltime; raise for long production runs
#SBATCH -p batch
#SBATCH -N 1                          # one node; one GPU is used per state point
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#
# Single state-point NPT density run on Frontier.
#
# Usage:
#   sbatch templates/submit_frontier.sbatch <output_dir> [extra run_npt.py flags]
# Example:
#   sbatch templates/submit_frontier.sbatch runs/flibe_783K \
#       --structure structure.pdb --model assets/mace_flibe.model --temperature 783.15
#
# Submit from the skill root so the relative paths below resolve.
set -euo pipefail

SKILL_ROOT="${SKILL_ROOT:-$(pwd)}"
cd "$SKILL_ROOT"

OUTPUT_DIR="${1:-runs/flibe}"
shift || true

source templates/frontier_env.sh

echo "Host: $(hostname)  |  ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"

# One MPI rank, one GPU, closest CPU cores. -c7 matches a Frontier L3-cache group.
srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -u scripts/run_npt.py --output-dir "$OUTPUT_DIR" "$@"
