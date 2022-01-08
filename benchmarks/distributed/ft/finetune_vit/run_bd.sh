#!/bin/bash

NNODES=3
NPROC_PER_NODE=4
MASTER_IP=192.168.64.11
MASTER_PORT=1234
export NCCL_SOCKET_IFNAME=bond0

ENABLE_LOGGING=${1:-0}
LOGGING_GROUP_SIZE=${2:-${NPROC_PER_NODE}}
PARALLEL_RECOVERY=${3:-0}

rm -rf *.h5
rm -rf *.log
rm -rf *.ckpt
rm -rf *.flag
hdfs dfs -rm -r "/*"

cmd="python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP \
	main.py \
	--micro-batch-size 32 \
	--global-batch-size 512 \
	--seed 42 \
	-p 1 \
	-j 32" 

LOGGING_ARGS="
	--logging \
	--logging-dfs hdfs \
	--logging-chunk-freq 5 \
	--logging-group-size ${LOGGING_GROUP_SIZE}"

if [[ $PARALLEL_RECOVERY -eq 1 ]]; then
	LOGGING_ARGS="${LOGGING_ARGS} --parallel-recovery"
fi

if [[ $ENABLE_LOGGING -eq 1 ]];then
	cmd="${cmd} ${LOGGING_ARGS}"
fi

cmd="${cmd} /data2/data/"

echo $cmd

OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 exec $cmd
