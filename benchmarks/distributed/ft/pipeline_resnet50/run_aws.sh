#!/bin/bash

NNODES=4
NPROC_PER_NODE=1
MASTER_IP=172.30.2.12
MASTER_PORT=1234

ENABLE_LOGGING=${1:-0}
LOGGING_GROUP_SIZE=${2:-${NPROC_PER_NODE}}
PARALLEL_RECOVERY=${3:-0}

rm -rf logging*.h5
rm -rf *.log
rm -rf *.ckpt
aws s3 rm s3://yczhong-swift/ --recursive --include='*.h5'

cmd="python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP \
	main.py \
	--micro-batch-size 16 \
	--global-batch-size 128 \
	--seed 2021 \
	-p 5 \
	-j 2" 

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

cmd="${cmd} ~/data/ILSVRC2012"

echo $cmd

OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
