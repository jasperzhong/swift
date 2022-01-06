#!/bin/bash

NNODES=2
NPROC_PER_NODE=2
MASTER_IP=10.28.1.16
MASTER_PORT=1234
export NCCL_SOCKET_IFNAME=eth2

DATA_PARALLEL_SIZE=${1:-1}
ENABLE_REPLICA=${2:-0}

rm -rf *.h5
rm -rf *.log
rm -rf *.ckpt
hdfs dfs -rm -r "/*"

cmd="python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP \
	main.py \
	--micro-batch-size 16 \
	--global-batch-size 128 \
	--seed 2021 \
	-p 1 \
	-j 4 \
	--data-parallel-size ${DATA_PARALLEL_SIZE}" 


if [[ $ENABLE_REPLICA -eq 1 ]];then
	cmd="${cmd} --replica"
fi

cmd="${cmd} ~/data/ILSVRC2012"

echo $cmd

OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
