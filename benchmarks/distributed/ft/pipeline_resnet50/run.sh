#!/bin/bash

NNODES=2
NPROC_PER_NODE=2
MASTER_IP=10.28.1.16
MASTER_PORT=1234
export NCCL_SOCKET_IFNAME=eth2

rm -rf logging.npy
pkill -f plasma_store

cmd="python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP \
	main.py \
	--micro-batch-size 32 \
	--global-batch-size 128 \
	--seed 2021 \
	-p 5 \
	--logging \
	~/data/ILSVRC2012" 

OMP_NUM_THREADS=8 NCCL_IB_DISABLE=1 LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
