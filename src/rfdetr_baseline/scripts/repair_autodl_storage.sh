#!/usr/bin/env bash
# Rebuild only the failed RF-DETR environment on the AutoDL data disk.
set -euo pipefail
if [[ "$(uname -s)" != Linux || ! -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  echo 'Run this only on the documented AutoDL Linux instance.' >&2
  exit 1
fi
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RFDETR_NEW_PREFIX=/root/autodl-tmp/conda-envs/rfdetr-nano
RFDETR_INSTALL_CACHE=/root/autodl-tmp/rfdetr-install-cache
export RFDETR_INSTALL_CACHE
export TMPDIR="$RFDETR_INSTALL_CACHE/tmp"
export PIP_CACHE_DIR="$RFDETR_INSTALL_CACHE/pip"
export CONDA_PKGS_DIRS="$RFDETR_INSTALL_CACHE/conda-pkgs"
mkdir -p "$TMPDIR" "$CONDA_PKGS_DIRS"
# Retain downloads from the failed attempt when the destination is new.
if [[ -d /root/.cache/pip && ! -e "$PIP_CACHE_DIR" && ! -L /root/.cache/pip ]]; then
  mv -- /root/.cache/pip "$PIP_CACHE_DIR"
fi
mkdir -p "$PIP_CACHE_DIR"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
python - <<'PY'
from pathlib import Path
import shutil
data = Path('/root/autodl-tmp').resolve()
free = shutil.disk_usage(data).free / 2**30
print(f'Data disk available: {free:.1f} GiB')
if free < 16:
    raise SystemExit('Need at least 16 GiB free on the data disk for installation; no environment was removed.')
old = Path('/root/miniconda3/envs/rfdetr-nano')
if old.is_symlink() or old.resolve().parent != Path('/root/miniconda3/envs').resolve():
    raise SystemExit('Old environment path differs from the documented target; stopped.')
PY
if [[ -d /root/miniconda3/envs/rfdetr-nano ]]; then
  conda env remove --prefix /root/miniconda3/envs/rfdetr-nano -y
fi
CONDA_PKGS_DIRS=/root/miniconda3/pkgs conda clean --tarballs --index-cache -y
cd "$TASK_ROOT"
if [[ ! -e "$RFDETR_NEW_PREFIX" ]]; then
  conda env create --prefix "$RFDETR_NEW_PREFIX" -f environment.yml
fi
conda activate "$RFDETR_NEW_PREFIX"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
python --version
python -c 'import sys, tempfile; print("Environment:", sys.prefix); print("Temporary directory:", tempfile.gettempdir())'
bash scripts/setup_autodl.sh
df -h / /root/autodl-tmp
echo 'Installation completed. In a new terminal activate /root/autodl-tmp/conda-envs/rfdetr-nano'
echo 'Next: bash scripts/acceptance.sh'
