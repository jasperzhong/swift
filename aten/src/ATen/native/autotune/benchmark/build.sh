set -e

export USE_CUDA=0
export USE_NNPACK=1
GIT_ROOT="$(git rev-parse --show-toplevel)"

pushd $GIT_ROOT > /dev/null
echo $(pwd)

mkdir -p build_libtorch && cd build_libtorch
python ../tools/build_libtorch.py
popd

cd "${GIT_ROOT}/aten/src/ATen/native/autotune/benchmark" > /dev/null
mkdir -p build
cd build

cmake -DCMAKE_PREFIX_PATH=$GIT_ROOT ..
cmake --build . --config Release
