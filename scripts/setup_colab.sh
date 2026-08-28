#!/bin/bash
# Setup script for Google Colab (also works on plain Linux with NVIDIA GPU).
#
# SSCMA (main branch) vendors mmengine and does not need mmcv/mmdet/mmcls,
# so no source compilation is required here. What matters:
#   1. torch/torchvision/torchaudio must be installed FIRST and must not be
#      replaced afterwards - pip's resolver does not respect locally
#      versioned (+cpu/+cuXXX) installed packages and would otherwise happily
#      replace them (e.g. when a dependency requires plain `torch`), which
#      can break the CUDA/driver match on Colab.
#   2. numpy is pinned (< 2.0) before anything else.
set -e

# always operate from the repository root, regardless of the caller's cwd
cd "$(dirname "${BASH_SOURCE[0]}")/.."


# ansi colors
RED='\033[031m'
GREEN='\033[032m'
BLUE='\033[034m'
RST='\033[m'


# check cuda
echo -en "Checking if CUDA available... "
if [ ! "$(command -v nvidia-smi)" ]; then
    echo -en "${RED}Not found!${RST}\n"
    echo -en "Please enable the GPU runtime (Runtime -> Change runtime type)${RST}\n"
    exit 1
else
    echo -en "${GREEN}OK${RST}\n"
fi


# step 1: ensure a modern pytorch stack; on Colab the preinstalled one is used
echo -en "${BLUE}Ensuring PyTorch stack... ${RST}\n"
if ! python -c "import torch, torchvision, torchaudio" > /dev/null 2>&1; then
    pip install torch torchvision torchaudio
fi
python -c "import torch, torchvision, torchaudio; print(f'torch={torch.__version__} torchvision={torchvision.__version__} torchaudio={torchaudio.__version__}')"

# pin the torch stack for every subsequent pip install (see header comment)
CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT
python -c "import torch, torchvision, torchaudio; print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}'); print(f'torchaudio=={torchaudio.__version__}')" > "${CONSTRAINTS_FILE}"
cat "${CONSTRAINTS_FILE}"


# step 2: pinned numpy and build helpers (TinyNeuralNetwork, used by
# tools/quantization.py, still relies on the distutils shim in setuptools)
echo -en "${BLUE}Installing build tools... ${RST}\n"
pip install "numpy>=1.23.0,<2.3.0" "setuptools>=64,<81" wheel


# step 3: python dependencies (with the torch stack constrained, nothing here
# can pull a different torch version)
echo -en "${BLUE}Installing dependencies... ${RST}\n"
pip install -c "${CONSTRAINTS_FILE}" -r requirements.txt


# step 4: install sscma itself (deps were installed above; --no-deps avoids
# re-resolving the torch stack)
echo -en "${BLUE}Installing sscma... ${RST}\n"
pip install --no-deps -e .


# step 5: smoke test
echo -en "${BLUE}Running smoke test... ${RST}\n"
python - <<'EOF'
import torch
import sscma
import sscma.datasets, sscma.models, sscma.engine, sscma.evaluation, sscma.visualization
from mmengine.runner.checkpoint import _torch_load

print(f'sscma={sscma.__version__} torch={torch.__version__}')
# checkpoints with non-tensor meta objects must load on PyTorch >= 2.6
import io
import numpy as np
buf = io.BytesIO()
torch.save({'state_dict': {}, 'meta': {'mean': np.array([0.5])}}, buf)
buf.seek(0)
_torch_load(buf)
print('checkpoint compatibility shim works: True')
EOF

echo -en "Finished setup... ${GREEN}OK${RST}\n"
