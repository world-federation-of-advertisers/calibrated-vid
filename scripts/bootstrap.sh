#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
VENV_DIR="${VENV_DIR:-.venv}"
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -e '.[notebook]'

if [[ "${VENV_DIR}" = /* ]]; then
  DISPLAY_VENV="${VENV_DIR}"
else
  DISPLAY_VENV="${PROJECT_DIR}/${VENV_DIR}"
fi
echo "Environment ready: ${DISPLAY_VENV}"
echo "Run tests with: ${VENV_DIR}/bin/python -m unittest discover -s tests -v"
echo "Open notebooks with: ${VENV_DIR}/bin/jupyter lab"
