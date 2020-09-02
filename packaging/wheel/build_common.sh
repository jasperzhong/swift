#!/usr/bin/env bash

set -eou pipefail
# SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
GIT_ROOT_DIR=$(git rev-parse --show-toplevel)
OUT_DIR=${OUT_DIR:-${GIT_ROOT_DIR}/out}

# # Function to retry functions that sometimes timeout or have flaky failures
# retry () {
#     $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
# }

PYTORCH_BUILD_VERSION=${PYTORCH_BUILD_VERSION:-0.0.1}
PYTORCH_BUILD_NUMBER=${PYTORCH_BUILD_NUMBER:-1}

# Include MKL in our CMAKE paths
# export CMAKE_LIBRARY_PATH="/opt/intel/lib:/lib:${CMAKE_LIBRARY_PATH}"
# export CMAKE_INCLUDE_PATH="/opt/intel/include:${CMAKE_INCLUDE_PATH}"

_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI:-1}
CMAKE_ARGS=${CMAKE_ARGS:-}
EXTRA_CAFFE2_CMAKE_FLAGS="${EXTRA_CAFFE2_CMAKE_FLAGS:-}"

(
    set -x
    python setup.py clean
    time \
        PYTORCH_BUILD_VERSION=${PYTORCH_BUILD_VERSION} \
        PYTORCH_BUILD_NUMBER=${PYTORCH_BUILD_NUMBER} \
        _GLIBCXX_USE_CXX11_ABI="${_GLIBCXX_USE_CXX11_ABI}" \
        CMAKE_ARGS="${CMAKE_ARGS[*]}" \
        EXTRA_CAFFE2_CMAKE_FLAGS="${EXTRA_CAFFE2_CMAKE_FLAGS[*]}" \
        python setup.py bdist_wheel -d "${OUT_DIR}"
)

# for pkg in "${OUT_DIR}"/torch*.whl; do
#     if [ -e "${pkg}" ]; then
#         echo "ERROR: ${pkg} is empty, did you actually build anything?"
#         exit 1
#     fi
#     tmp_dir=$(mktemp -d)
#     trap 'rm -rfv ${tmp_dir}' EXIT
#     (
#         set -x
#         cp "${pkg}" "${tmp_dir}"
#     )
# done
