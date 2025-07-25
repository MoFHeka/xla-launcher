workspace(name = "xla_launcher")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#
# Setup XLA.
#
xla_hash = "3984312b5a54af47f9f12737fc257ce7e4aff7c8"  # Committed on 2025-07-25.
# xla_hash = "main"

http_archive(
    name = "xla",
    patch_args = [
        "-l",
        "-p1",
    ],
    # Patch XLA profiler visibility
    patch_cmds = [
        """
    sed -i 's/visibility = \\["\\/\\/xla:internal"\\]/visibility = ["\\/\\/visibility:public"]/g' xla/backends/profiler/plugin/BUILD
    """,
        """
    sed -i 's/visibility = \\["\\/\\/xla:friends"\\]/visibility = ["\\/\\/visibility:public"]/g' xla/tools/BUILD
    """,
    ],
    patch_tool = "patch",
    patches = [
        # "//xla_patches:gpu_nvml.diff",  # No needed if use hermetic cuda package
        "//xla_patches:gpu_race_condition.diff",  # TODO(He Jia): Is it needed?
        "//xla_patches:count_down.diff",
    ],
    strip_prefix = "xla-" + xla_hash,
    urls = ["https://github.com/openxla/xla/archive/" + xla_hash + ".tar.gz"],
)

# Initialize the XLA repository and all dependencies.
#
# The cascade of load() statements and xla_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.

load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

# Please run
# python3 -c "import sys; print(f'pip-compile build_deps/python/requirements.in --output-file=build_deps/python/requirements_lock_{sys.version_info.major}_{sys.version_info.minor}.txt')" | sh
# firstly to generate the requirements_lock_*.txt files
# You can also use the following command to install the dependencies according to your python version:
# pip-compile build_deps/python/requirements.in --output-file=build_deps/python/requirements_lock_3_10.txt
# pip-compile build_deps/python/requirements.in --output-file=build_deps/python/requirements_lock_3_11.txt
# pip-compile build_deps/python/requirements.in --output-file=build_deps/python/requirements_lock_3_12.txt
# pip-compile build_deps/python/requirements.in --output-file=build_deps/python/requirements_lock_3_13.txt
python_init_repositories(
    requirements = {
        "3.10": "//build_deps/python:requirements_lock_3_10.txt",
        "3.11": "//build_deps/python:requirements_lock_3_11.txt",
        "3.12": "//build_deps/python:requirements_lock_3_12.txt",
        "3.13": "//build_deps/python:requirements_lock_3_13.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

load(
    "@rules_ml_toolchain//cc_toolchain/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc_toolchain:lx64_lx64")

register_toolchains("@rules_ml_toolchain//cc_toolchain:lx64_lx64_cuda")

load(
    "@rules_ml_toolchain//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load(
    "@rules_ml_toolchain//third_party/nvshmem/hermetic:nvshmem_json_init_repository.bzl",
    "nvshmem_json_init_repository",
)

nvshmem_json_init_repository()

load(
    "@nvshmem_redist_json//:distributions.bzl",
    "NVSHMEM_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//third_party/nvshmem/hermetic:nvshmem_redist_init_repository.bzl",
    "nvshmem_redist_init_repository",
)

nvshmem_redist_init_repository(
    nvshmem_redistributions = NVSHMEM_REDISTRIBUTIONS,
)

http_archive(
    name = "dlpack_latest",
    build_file_content = """
cc_library(
    name = "dlpack",
    hdrs = ["include/dlpack/dlpack.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
    strip_prefix = "dlpack-1.1",
    urls = ["https://github.com/dmlc/dlpack/archive/refs/tags/v1.1.zip"],
)
