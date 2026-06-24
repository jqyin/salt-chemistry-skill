#!/bin/bash
# Interactive single-GPU launcher (modernized successor to the original).
#
# Sets up the Frontier environment and runs one NPT state point on the GPU you
# are placed on. Intended to be launched *under* srun for an interactive node,
# mirroring the original workflow:
#
#   chmod +x scripts/run_openmm_onegpu.sh
#   PRODUCTION_STEPS=1000 EQUILIBRATION_STEPS=0 \
#     srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
#     ./scripts/run_openmm_onegpu.sh --output-dir runs/smoke
#
# Any extra arguments are passed straight through to scripts/run_npt.py, so you
# can also drive it fully by flags:
#   srun ... ./scripts/run_openmm_onegpu.sh --temperature 900 --production-steps 250000
set -euo pipefail

SKILL_ROOT="${SKILL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SKILL_ROOT"

source templates/frontier_env.sh

echo "Running on host: $(hostname)"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-not_set}"
echo "OPENMM_PLATFORM=$OPENMM_PLATFORM  STEPS(prod)=${PRODUCTION_STEPS:-${STEPS:-default}}"

# Quick GPU sanity check before the (expensive) MACE setup.
python - <<'PY'
import torch
print("torch:", torch.__version__, "| hip:", torch.version.hip, "| cuda:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available(), "| count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
PY

python -u scripts/run_npt.py "$@"
