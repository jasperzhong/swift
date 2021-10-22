import argparse
import os
from datetime import timedelta

import torch

parser = argparse.ArgumentParser("all_reduce")
parser.add_argument("--master_ip", type=str, default=None)


def main():
    args = parser.parse_args()

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    master_port = int(os.environ['MASTER_PORT'])
    init_method = "tcp://{}:{}".format(args.master_ip, master_port)
    torch.distributed.init_process_group(
        'nccl', init_method=init_method,
        world_size=world_size, rank=rank,
        timeout=timedelta(seconds=10)
    )

    x = torch.randn((1000)).cuda()

    while True:
        torch.distributed.all_reduce(x)


if __name__ == '__main__':
    main()
