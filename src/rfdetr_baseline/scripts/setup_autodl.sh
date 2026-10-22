#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
# Large downloads and extraction must not use the small system disk.
RFDETR_INSTALL_CACHE="${RFDETR_INSTALL_CACHE:-/root/autodl-tmp/rfdetr-install-cache}"
export TMPDIR="${TMPDIR:-$RFDETR_INSTALL_CACHE/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$RFDETR_INSTALL_CACHE/pip}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
python - <<'PY'
import sys, platform
if platform.system() != 'Linux' or sys.version_info[:2] != (3, 11):
    raise SystemExit('Use the isolated Linux Python 3.11 conda environment described in README.md.')
print('Python:', sys.version)
PY
mkdir -p verification
python -m pip install pip==25.0.1 setuptools==75.8.0 wheel==0.45.1
python -m pip install torch==2.4.1 torchvision==0.19.1 -c constraints.txt --index-url https://download.pytorch.org/whl/cu121
python -m pip install --require-hashes -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e vendor/rf_detr
python -m pip check
python -m pip freeze > verification/environment.freeze.txt
python verify_env.py --strict --output verification/environment.json
echo 'Environment installed. Next: python smoke_test.py --steps 3'
