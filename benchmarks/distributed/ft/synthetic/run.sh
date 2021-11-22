#!/bin/bash

NNODES=2
NPROC_PER_NODE=2
MASTER_IP=10.28.1.16
export NCCL_SOCKET_IFNAME=eth2

cmd="python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP \
	pytorch_synthetic_benchmark.py"

NCCL_IB_DISABLE=1 LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
