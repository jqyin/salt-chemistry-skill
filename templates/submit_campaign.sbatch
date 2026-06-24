#!/bin/bash
#SBATCH -A <YOUR_PROJECT_ID>
#SBATCH -J flibe_campaign
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1                          # one node per array task = one GPU per state point
#SBATCH -o %x-%A_%a.out
#SBATCH -e %x-%A_%a.err
#
# Run a whole temperature x composition campaign as a SLURM job array. Each array
# task picks one run directory (built by scripts/make_campaign.py) and runs it.
#
# Usage:
#   python scripts/make_campaign.py --temperatures 700,800,900 --mol-percent-bef2 33.33 --base-dir campaign
#   sbatch --array=0-$(($(wc -l < campaign/run_dirs.txt)-1)) \
#       templates/submit_campaign.sbatch campaign/run_dirs.txt
#
# Submit from the skill root.
set -euo pipefail

SKILL_ROOT="${SKILL_ROOT:-$(pwd)}"
cd "$SKILL_ROOT"

RUN_DIRS_FILE="${1:?Pass the run_dirs.txt produced by make_campaign.py}"

source templates/frontier_env.sh

# Pick this array task's run directory (0-indexed -> sed line is 1-indexed).
LINE=$((SLURM_ARRAY_TASK_ID + 1))
RUN_DIR="$(sed -n "${LINE}p" "$RUN_DIRS_FILE")"
if [[ -z "$RUN_DIR" ]]; then
    echo "No run dir at line $LINE of $RUN_DIRS_FILE" >&2
    exit 1
fi
echo "Array task $SLURM_ARRAY_TASK_ID -> $RUN_DIR on $(hostname)"

# Each run dir holds its own structure.pdb + config.yaml; run there.
srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -u scripts/run_npt.py --config "$RUN_DIR/config.yaml" --output-dir "$RUN_DIR"
