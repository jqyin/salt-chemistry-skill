#!/bin/bash
# Frontier (OLCF) environment setup for the OpenMM + MACE stack.
#
# Source this from a batch script or interactive shell:  source templates/frontier_env.sh
# It loads the ROCm/PrgEnv modules and activates the conda env, creating the env
# from the YAML on first use. Override module/env versions via the variables below
# if OLCF updates the software stack.
set -euo pipefail

PRGENV_MODULE="${PRGENV_MODULE:-PrgEnv-gnu/8.7.0}"
CPE_MODULE="${CPE_MODULE:-cpe/26.03}"
MINIFORGE_MODULE="${MINIFORGE_MODULE:-miniforge3/23.11.0-0}"
ROCM_MODULE="${ROCM_MODULE:-rocm/7.1.1}"
ACCEL_MODULE="${ACCEL_MODULE:-craype-accel-amd-gfx90a}"

# Path to the conda environment YAML (relative to wherever you source this from).
ENV_YML="${ENV_YML:-environment.frontier.yml}"

module purge
module load "$PRGENV_MODULE"
module load "$CPE_MODULE"
module load "$MINIFORGE_MODULE"
module load "$ROCM_MODULE"
module load "$ACCEL_MODULE"

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

eval "$(conda shell.bash hook)"

ENV_NAME="${ENV_NAME:-$(awk -F': *' '/^name:/ {print $2; exit}' "$ENV_YML")}"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating conda env '$ENV_NAME' from $ENV_YML (first-time setup)..."
    conda env create -f "$ENV_YML"
fi
conda activate "$ENV_NAME"

# OpenMM backend selection for the AMD MI250X GPUs on Frontier.
export OPENMM_PLATFORM="${OPENMM_PLATFORM:-HIP}"
export OPENMM_PRECISION="${OPENMM_PRECISION:-mixed}"
export OPENMM_DEVICE_INDEX="${OPENMM_DEVICE_INDEX:-0}"

echo "Environment ready: $ENV_NAME on $(hostname) | OPENMM_PLATFORM=$OPENMM_PLATFORM"
