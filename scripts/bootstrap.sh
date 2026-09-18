#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,notebook]'
.venv/bin/python -m grpc_tools.protoc \
  --proto_path=proto \
  --python_out=src \
  proto/calibrated_vid/synthetic_campaign_experiment.proto
