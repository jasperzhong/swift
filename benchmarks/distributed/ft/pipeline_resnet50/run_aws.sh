#!/bin/bash

NNODES=4
NPROC_PER_NODE=1
MASTER_IP=172.30.2.12
MASTER_PORT=1234


DATA_PARALLEL_SIZE=${1:-1}
ENABLE_REPLICA=${2:-0}

rm -rf *.h5
rm -rf *.log
rm -rf *.ckpt
aws s3 rm s3://yczhong-swift/ --recursive --include='*.h5'

cmd="python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP \
	main.py \
	--micro-batch-size 32 \
	--global-batch-size 512 \
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
