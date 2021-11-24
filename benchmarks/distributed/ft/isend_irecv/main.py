import os
from datetime import timedelta

import torch



@torch.distributed.fault_tolerance.run
def train(state):
    print("start from i={}".format(state.i))
    rank = torch.distributed.get_rank()
    size = (1000000, )
    while True:
        dst_rank = (rank + 1) % 2
        x = torch.ones(size=size, requires_grad=True).cuda()
        y = torch.zeros(size=size, requires_grad=True).cuda()
        send_op = torch.distributed.P2POp(torch.distributed.isend, x,
                                          dst_rank)
        recv_op = torch.distributed.P2POp(torch.distributed.irecv, y,
                                          dst_rank)
        reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()

        torch.cuda.synchronize()
        z = x + y

        state.i += 1
        if state.i % 1000 == 0:
            print(state.i)


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl', init_method="env://",
        timeout=timedelta(seconds=5)
    )

    state = torch.distributed.fault_tolerance.State(i=0)
    train(state)

if __name__ == '__main__':
    main()
