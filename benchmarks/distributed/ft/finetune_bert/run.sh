#!/bin/bash

NNODES=2
NPROC_PER_NODE=4
MASTER_IP=10.28.1.27
MASTER_PORT=1234
export NCCL_SOCKET_IFNAME=enp94s0

ENABLE_LOGGING=${1:-0}
LOGGING_GROUP_SIZE=${2:-${NPROC_PER_NODE}}
PARALLEL_RECOVERY=${3:-0}

rm -rf *.h5
rm -rf *.log
rm -rf *.ckpt
hdfs dfs -rm -r "/*"

cmd="python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP \
	main.py \
	--micro-batch-size 4 \
	--global-batch-size 32 \
	--test-batch-size 157 \
	--seed 42 \
	-p 5 \
	--do_lower_case \
	/home/$USER/data/squad/v1.1/train-v1.1.json \
	--predict-file /home/$USER/data/squad/v1.1/dev-v1.1.json
	--vocab_file ./vocab"

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

export HADOOP_MASTER=10.28.1.27

OMP_NUM_THREADS=8 NCCL_IB_DISABLE=1 exec $cmd
