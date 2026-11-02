#!/usr/bin/env bash
set -euo pipefail
cd /workspace/polar
python3 -m venv --system-site-packages /workspace/.polar-venv
export PIP_CACHE_DIR=/workspace/.polar-setup/pip-cache
export MPLCONFIGDIR=/workspace/.polar-setup/matplotlib
export XDG_CACHE_HOME=/workspace/.polar-setup/cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
mkdir -p /home/agent/.drjit
/workspace/.polar-venv/bin/python -m pip install 'torch==2.12.1' --index-url https://download.pytorch.org/whl/cpu
/workspace/.polar-venv/bin/python -m pip install 'numpy==2.3.5' 'scipy==1.17.0' 'matplotlib==3.10.8'
python3 -m venv --system-site-packages /workspace/.polar-phase1-venv
/workspace/.polar-phase1-venv/bin/python -m pip install 'tensorflow==2.21.0' 'sionna-rt==2.0.1'
cp scripts/cloud/run-phase1.py /workspace/.polar-setup/run-phase1.py
OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /workspace/.polar-phase1-venv/bin/python -c "import runpy; runpy.run_path('/workspace/.polar-setup/run-phase1.py'); print('Phase 1 imports OK')"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg /workspace/.polar-venv/bin/python fec/polar/test_crypto.py
