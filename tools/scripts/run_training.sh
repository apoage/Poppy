#!/bin/bash
# Wrapper script to run training with correct CUDA library paths

# Get project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate virtual environment
source "$PROJECT_ROOT/trader_env/bin/activate"

# Get site-packages path
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Set LD_LIBRARY_PATH to use venv's CUDA libraries (CuDNN 9.8+)
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_nvrtc/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/curand/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/nccl/lib:$LD_LIBRARY_PATH"

echo "✅ CUDA libraries configured from venv"
echo "   CuDNN: $SITE_PACKAGES/nvidia/cudnn/lib"

# Run whatever command was passed
exec "$@"
