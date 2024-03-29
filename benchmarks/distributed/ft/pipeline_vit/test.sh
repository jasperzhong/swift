#!/bin/bash

NNODES=1
NPROC_PER_NODE=1
MASTER_IP=10.28.1.27
MASTER_PORT=1234
export NCCL_SOCKET_IFNAME=enp94s0

rm -rf logging*.h5

cmd="python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP \
	ViTTest.py \
	--micro-batch-size 32 \
	--global-batch-size 128 \
	--seed 2021 \
	-p 5 \
	--logging \
	~/data/ILSVRC2012" 

OMP_NUM_THREADS=8 NCCL_IB_DISABLE=1 LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
