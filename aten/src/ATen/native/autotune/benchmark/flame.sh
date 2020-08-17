#! /bin/bash
set -e

OUTPUT="/tmp/taylorrobie/performance"
SCRATCH="${OUTPUT}/scratch"
mkdir -p $SCRATCH

function flame {
    IMPL=$1
    C_IN=$2
    C_OUT=$3
    echo "${IMPL}  ${C_IN}  ${C_OUT}"

    FLAME="${OUTPUT}/FlameGraph"
    git clone "https://github.com/brendangregg/FlameGraph" "$FLAME" 2> /dev/null || (cd "$FLAME" ; git pull)

    perf record -g -o "${SCRATCH}/perf.data" -F 500 ./build/benchmark_autotune profile $IMPL $C_IN $C_OUT

    pushd $SCRATCH
    perf script > out.parsed
    "${FLAME}/stackcollapse-perf.pl" out.parsed > out.perf-folded
    "${FLAME}/flamegraph.pl" out.perf-folded > "/mnt/shared/taylorrobie/public_html/conv_prof_${IMPL}_${C_IN}_${C_OUT}.svg"
    popd

    echo
}

flame Native 1,1024,1024,4,4
flame NNPack 1,1024,1024,4,4
flame MKL 1,1024,1024,4,4


# flame Native 1 128
# flame Native 128 1
# flame Native 32 32
# flame Native 64 64

# flame NNPack 1 128
# flame NNPack 128 1
# flame NNPack 32 32
# flame NNPack 64 64

# flame MKL 1 128
# flame MKL 128 1
# flame MKL 32 32
# flame MKL 64 64
