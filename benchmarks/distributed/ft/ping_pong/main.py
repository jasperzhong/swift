import argparse
import os
from datetime import timedelta

import torch

parser = argparse.ArgumentParser("ping pong")
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
        'gloo', init_method=init_method,
        world_size=world_size, rank=rank,
        timeout=timedelta(seconds=5)
    )

    x = torch.randn((100000))

    i = 0
    while True:
        src_rank = i % world_size
        dst_rank = (i + 1) % world_size
        if torch.distributed.get_rank() == src_rank:
            torch.distributed.send(x, dst_rank)
        elif torch.distributed.get_rank() == dst_rank:
            torch.distributed.recv(x, src_rank)
        i += 1
        if i % 1000 == 0:
            print(i)


if __name__ == '__main__':
    main()
