#!/usr/bin/env bash
# Rebuild the A-server boxpromptiml environment on B.
# This script is a convenience wrapper; review it before running.
# It does NOT download datasets or model weights.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-boxpromptiml}"
CUDA_TAG="${CUDA_TAG:-cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.2.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.17.1}"
SEGMENT_ANYTHING_SRC="${SEGMENT_ANYTHING_SRC:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not on PATH. Install Miniconda or adapt this script." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
ENV_PREFIX="$CONDA_BASE/envs/$CONDA_ENV"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  echo "==> creating conda env $CONDA_ENV (python 3.10)"
  conda create -y -n "$CONDA_ENV" python=3.10 pip
else
  echo "==> reusing existing conda env $CONDA_ENV"
fi

echo "==> installing PyTorch ${TORCH_VERSION}+${CUDA_TAG}"
"$ENV_PREFIX/bin/pip" install \
  "torch==${TORCH_VERSION}+${CUDA_TAG}" \
  "torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> installing remaining pinned dependencies"
"$ENV_PREFIX/bin/pip" install -r "$HERE/pip-requirements.txt"

echo "==> installing segment-anything"
if [ -n "$SEGMENT_ANYTHING_SRC" ]; then
  "$ENV_PREFIX/bin/pip" install "$SEGMENT_ANYTHING_SRC"
else
  "$ENV_PREFIX/bin/pip" install "git+https://github.com/facebookresearch/segment-anything.git"
fi

echo "==> checking the environment"
"$ENV_PREFIX/bin/python" - <<'PY'
import sys
import torch
import torchvision
import numpy
import timm
import cv2
print("python    ", sys.version.split()[0])
print("torch     ", torch.__version__)
print("torchvision", torchvision.__version__)
print("torch.cuda", torch.version.cuda, "available:", torch.cuda.is_available(),
      "devices:", torch.cuda.device_count())
print("numpy     ", numpy.__version__)
print("timm      ", timm.__version__)
print("cv2       ", cv2.__version__)
PY
"$ENV_PREFIX/bin/pip" check

cat <<EOF

Environment ready: $ENV_PREFIX
Next steps:
  1. Restore datasets/weights/runs and create the /hy-tmp path aliases (see MIGRATION.md).
  2. Put the TinyViT checkpoint into the torch hub cache (see MIGRATION.md).
  3. Run: $ENV_PREFIX/bin/python migration/verify_paths.py
EOF
