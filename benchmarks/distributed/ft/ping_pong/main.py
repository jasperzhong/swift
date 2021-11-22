import argparse
import os
from datetime import timedelta

import torch

parser = argparse.ArgumentParser("ping pong")

@torch.distributed.run
def train(ts, world_size):
   # ts.sync()
    x = torch.randn((100000)).cuda()
    print("start from i={}".format(ts.i))
    while True:
        src_rank = ts.i % world_size
        dst_rank = (ts.i + 1) % world_size
        if torch.distributed.get_rank() == src_rank:
            rc = torch.distributed.send(x, dst_rank)
        elif torch.distributed.get_rank() == dst_rank:
            rc = torch.distributed.recv(x, src_rank)
        if rc is False:
            continue
        ts.i += 1
        if ts.i % 1000 == 0:
            print(ts.i)

def main():
    args = parser.parse_args()

    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl', init_method="env://",
        timeout=timedelta(seconds=5)
    )

    ts = torch.distributed.TimeStamp(i=0)
    train(ts, world_size)


if __name__ == '__main__':
    main()
