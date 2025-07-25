# PY_VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
# python configure.py \
#   --backend CUDA \
#   --local_cuda_path "$CUDA_PATH" \
#   --local_cudnn_path "$CONDA_PREFIX/lib/python$PY_VER/site-packages/nvidia/cudnn/" \
#   --local_nccl_path "$CONDA_PREFIX/lib/python$PY_VER/site-packages/nvidia/nccl/"

# Only Clang in HERMETIC_CC_TOOLCHAIN
# Recommend to use clang-17+
python configure.py \
--backend CUDA \
--host_compiler clang \
--cuda_compiler clang \
--cuda_version 12.8.1 \
--cudnn_version 9.8.0

# If you want to use nvcc for building cuda, set --config=clang_local and make USE_HERMETIC_CC_TOOLCHAIN=0
# bazel --output_base=./bazel_output build //xla_launcher:xla_launcher_tar_gz
bazel build //xla_launcher:xla_launcher_tar_gz
