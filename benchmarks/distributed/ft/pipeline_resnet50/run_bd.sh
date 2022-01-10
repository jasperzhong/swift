#!/bin/bash

NNODES=16
NPROC_PER_NODE=4
MASTER_IP=192.168.64.11
MASTER_PORT=1234
export NCCL_SOCKET_IFNAME=bond0

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
	--global-batch-size 4096 \
	--benchmark-iters 200 \
	--seed 42 \
	-p 1 \
	-j 32 \
	--data-parallel-size ${DATA_PARALLEL_SIZE}" 


if [[ $ENABLE_REPLICA -eq 1 ]];then
	cmd="${cmd} --replica"
fi


cmd="${cmd} /data2/data/ILSVRC2012"

echo $cmd

OMP_NUM_THREADS=4 NCCL_IB_DISABLE=1 exec $cmd
