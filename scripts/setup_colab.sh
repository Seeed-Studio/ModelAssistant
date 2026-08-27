#!/bin/bash
# Setup script for Google Colab (also works on plain Linux with NVIDIA GPU).
#
# Order matters:
#   1. torch/torchvision/torchaudio must be installed FIRST and must not be
#      replaced afterwards - the mmcv extension is compiled against them.
#      (Some dependencies, e.g. pyvww, pull an unpinned `torchvision`, whose
#      latest release pins an exact torch version and would silently swap it.)
#   2. numpy is pinned (< 2.0) BEFORE anything is compiled.
#   3. mmcv is compiled from source LAST, after all pip installs are done.
set -e


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

# Pin the torch stack for every subsequent pip install: pip's resolver does
# not respect locally-versioned (+cpu/+cuXXX) installed packages and would
# otherwise happily replace them (e.g. pnnx depends on plain `torch`), which
# breaks the mmcv build and the CUDA/driver match.
python -c "import torch, torchvision, torchaudio; print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}'); print(f'torchaudio=={torchaudio.__version__}')" > /tmp/sscma_constraints.txt
cat /tmp/sscma_constraints.txt


# step 2: build tools and pinned numpy (must happen before compiling mmcv)
echo -en "${BLUE}Installing build tools... ${RST}\n"
# setuptools<81: still ships the distutils shim required by TinyNeuralNetwork
pip install "numpy>=1.23.0,<2.0.0" "setuptools>=49.4.0,<81" Cython ninja wheel packaging


# step 3: python dependencies (with the torch stack constrained, nothing here
# can pull a different torch version)
echo -en "${BLUE}Installing base deps... ${RST}\n"
pip install -c /tmp/sscma_constraints.txt -r requirements/base.txt -r requirements/inference.txt -r requirements/export.txt -r requirements/tests.txt


# step 4: OpenMMLab deps (pip metadata of mmdet/mmcls does not pull mmcv, so
# this cannot interfere with the mmcv source build below)
echo -en "${BLUE}Installing OpenMMLab deps... ${RST}\n"
pip install -c /tmp/sscma_constraints.txt "mmengine>=0.8.2,<1.0.0" "mmdet>=3.0.0,<3.1.0" "mmcls>=1.0.0rc6"


# step 5: build mmcv from source - there are no prebuilt wheels for recent
# PyTorch/Python versions. Compiling with CUDA ops takes ~10-15 min on Colab.
echo -en "${BLUE}Building mmcv 2.2.0 from source (this takes a while)... ${RST}\n"
MMCV_WITH_OPS=1 MAX_JOBS="$(nproc)" pip install --no-build-isolation --no-cache-dir --no-binary mmcv "mmcv==2.2.0"


# step 6: relax the mmcv < 2.1.0 assertion hard-coded in mmdet/mmcls
echo -en "${BLUE}Patching OpenMMLab version checks... ${RST}\n"
python scripts/patch_mmlab_versions.py --verify


# step 7: install sscma itself (deps were installed above; --no-deps avoids
# re-resolving legacy pins)
echo -en "${BLUE}Installing sscma... ${RST}\n"
pip install --no-deps -e .


# step 8: smoke test
echo -en "${BLUE}Running smoke test... ${RST}\n"
python - <<'EOF'
import torch
import mmcv
import mmdet
import mmcls
import mmengine
import sscma.datasets, sscma.engine, sscma.evaluation, sscma.models, sscma.visualization
from mmcv.ops import nms

keep = nms(torch.rand(10, 4), torch.rand(10), 0.5)
print(f'sscma={sscma.__version__} mmcv={mmcv.__version__} mmdet={mmdet.__version__} '
      f'mmcls={mmcls.__version__} mmengine={mmengine.__version__} torch={torch.__version__}')
print('mmcv ops (nms) work:', len(keep[0]) > 0)
EOF

echo -en "Finished setup... ${GREEN}OK${RST}\n"
