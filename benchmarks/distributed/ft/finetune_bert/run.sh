#!/bin/bash

NNODES=2
NPROC_PER_NODE=4
MASTER_IP=10.28.1.27
MASTER_PORT=1234
export NCCL_SOCKET_IFNAME=enp94s0

cmd="python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP \
	main.py \
	--micro-batch-size 16 \
	--global-batch-size 128 \
	--test-batch-size 128 \
	--seed 2021 \
	-p 5 \
	--do_lower_case \
	/home/gmsheng/data/squad/v1.1/train-v1.1.json \
	--predict-file /home/gmsheng/data/squad/v1.1/dev-v1.1.json" 

OMP_NUM_THREADS=8 NCCL_IB_DISABLE=1 LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
