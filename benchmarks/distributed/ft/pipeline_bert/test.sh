NNODES=1
NPROC_PER_NODE=1
MASTER_IP=192.168.64.18
MASTER_PORT=1234
export NCCL_SOCKET_IFNAME=bond0

cmd="python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP \
	test.py \
	--micro-batch-size 4 \
	--global-batch-size 512 \
	--seed 2021 \
	-p 5 \
	/data2/data/BERT" 

OMP_NUM_THREADS=8 NCCL_IB_DISABLE=1 LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
