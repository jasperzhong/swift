import argparse
import os
import time
from datetime import timedelta

import torch

parser = argparse.ArgumentParser("all_reduce")

@torch.distributed.run
def train(ts):
    print("start from i={}".format(ts.i))
    x = torch.randn((32, 128, 1024)).cuda()
    y = torch.randn((1024, 1024)).cuda()

    while True:
        x = x @ y
        torch.distributed.all_reduce(y)
        x = torch.relu(x)
        ts.i += 1
        if ts.i % 1000 == 0:
            print(ts.i)


def main():
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl', init_method="env://",
        timeout=timedelta(seconds=5)
    )

    ts = torch.distributed.TimeStamp(i=0)
    train(ts)



if __name__ == '__main__':
    main()
