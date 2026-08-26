#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
VENV_DIR="${VENV_DIR:-.venv}"
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -e '.[notebook]'

echo "Environment ready: ${PROJECT_DIR}/${VENV_DIR}"
echo "Run tests with: ${VENV_DIR}/bin/python -m unittest discover -s tests -v"
echo "Open notebooks with: ${VENV_DIR}/bin/jupyter lab"
