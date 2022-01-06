#!/bin/bash

NNODES=2
NPROC_PER_NODE=4
MASTER_IP=10.28.1.27
MASTER_PORT=1234
export NCCL_SOCKET_IFNAME=enp94s0

rm -rf *.h5
rm -rf *.log
rm -rf *.ckpt

cmd="python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP \
	main.py \
	--micro-batch-size 4 \
	--global-batch-size 32 \
	--seed 2021 \
	-p 5 \
	--benchmark-iters 300 \
	/home/gmsheng/data/bert" 

OMP_NUM_THREADS=8 NCCL_IB_DISABLE=1 LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
