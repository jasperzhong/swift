import argparse
import os
from datetime import timedelta

import torch
import torch.distributed.fault_tolerance

parser = argparse.ArgumentParser("ping pong")


@torch.distributed.fault_tolerance.run
def train(state, world_size):
    print("start from i={}".format(state.i))

    x = torch.randn((100000)).cuda()
    reqs = []
    while True:
        src_rank = state.i % world_size
        dst_rank = (state.i + 1) % world_size
        if torch.distributed.get_rank() == src_rank:
            req = torch.distributed.isend(x, dst_rank)
        elif torch.distributed.get_rank() == dst_rank:
            req = torch.distributed.irecv(x, src_rank)
        reqs.append(req)

        state.i += 1
        if state.i % 1000 == 0:
            for req in reqs:
                req.wait()
            reqs.clear()
            print(state.i)


def main():
    args = parser.parse_args()

    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl', init_method="env://",
        timeout=timedelta(seconds=5)
    )

    state = torch.distributed.fault_tolerance.State(i=0)
    train(state, world_size)


if __name__ == '__main__':
    main()
