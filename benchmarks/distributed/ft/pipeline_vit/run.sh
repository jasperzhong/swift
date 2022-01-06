#!/bin/bash

NNODES=2
NPROC_PER_NODE=2
MASTER_IP=10.28.1.16
MASTER_PORT=1234
export NCCL_SOCKET_IFNAME=eth2

rm -rf logging*.h5

ENABLE_LOGGING=${1:-0}
LOGGING_GROUP_SIZE=${2:-${NPROC_PER_NODE}}
PARALLEL_RECOVERY=${3:-0}

rm -rf *.h5
rm -rf *.log
rm -rf *.ckpt
hdfs dfs -rm -r "/*"

cmd="python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP \
	main.py \
	--micro-batch-size 4 \
	--global-batch-size 32 \
	--seed 2021 \
	-p 1 \
	-j 4" 

LOGGING_ARGS="
	--logging \
	--logging-dfs s3 \
	--logging-s3-bucket yczhong-swift \
	--logging-group-size ${LOGGING_GROUP_SIZE}"

if [[ $PARALLEL_RECOVERY -eq 1 ]]; then
	LOGGING_ARGS="${LOGGING_ARGS} --parallel-recovery"
fi

if [[ $ENABLE_LOGGING -eq 1 ]];then
	cmd="${cmd} ${LOGGING_ARGS}"
fi

cmd="${cmd} ~/data/ILSVRC2012_ImageNet"

echo $cmd

OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
