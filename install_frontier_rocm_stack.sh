#!/bin/bash
set -euo pipefail

module purge
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
if [[ -n "${CRAY_LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "openmm-torch-frontier"; then
    echo "Environment openmm-torch-frontier already exists."
    echo "Remove it first with:"
    echo "conda env remove -n openmm-torch-frontier -y"
    exit 1
fi

conda env create -f openmm-torch-frontier.yml
conda activate openmm-torch-frontier

python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/rocm7.1
python -m pip install mace-torch==0.3.16 e3nn==0.4.4 torch-ema==0.3 torchmetrics lmdb matscipy prettytable orjson python-hostlist
python -m pip install 'openmm[hip7]' openmmml
python -m ipykernel install --user --name openmm-torch-frontier --display-name 'Python (openmm-torch-frontier)'
