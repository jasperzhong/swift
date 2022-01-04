import torch

_GLOBAL_ARGS = None

_cnt = 0


def initialize_global_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def is_pipeline_last_stage():
    return get_pipeline_model_parallel_rank() == \
        get_pipeline_model_parallel_world_size() - 1


def is_pipeline_first_stage():
    return get_pipeline_model_parallel_rank() == 0


def get_pipeline_model_parallel_world_size():
    return torch.distributed.get_world_size()


def get_pipeline_model_parallel_rank():
    global _GLOBAL_ARGS
    return torch.distributed.get_rank()  // _GLOBAL_ARGS.data_parallel_size


def get_pipeline_model_parallel_next_rank():
    return (get_pipeline_model_parallel_rank() + 1) % \
        get_pipeline_model_parallel_world_size()


def get_pipeline_model_parallel_prev_rank():
    return (get_pipeline_model_parallel_rank() - 1) % \
        get_pipeline_model_parallel_world_size()


def get_num_microbatches():
    global _GLOBAL_ARGS
    return _GLOBAL_ARGS.global_batch_size // _GLOBAL_ARGS.micro_batch_size // \
        torch.distributed.parallel_recovery_data_parallel_size()


def get_microbatch_size():
    global _GLOBAL_ARGS
    return _GLOBAL_ARGS.micro_batch_size


def forward_step(data_iterator, model, input_tensor, loss_func, loss):
    if is_pipeline_first_stage() or is_pipeline_last_stage():
        data = next(data_iterator)
        images, labels = data

        if is_pipeline_first_stage():
            images = images.cuda()
        elif is_pipeline_last_stage():
            labels = labels.cuda()

    if is_pipeline_first_stage():
        assert input_tensor is None
        input_tensor = images

    output_tensor = model(input_tensor)

    if is_pipeline_last_stage():
        output_tensor = loss_func(output_tensor, labels)
        output_tensor /= get_num_microbatches()
        loss += output_tensor.item()

    return output_tensor


def backward_step(input_tensor, output_tensor, output_tensor_grad):
    global _cnt
    if input_tensor is not None:
        input_tensor.retain_grad()

    torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)

    input_tensor_grad = None
    if input_tensor is not None:
        input_tensor_grad = input_tensor.grad
        
    return input_tensor_grad


def send_forward(output_tensor):
    if not is_pipeline_last_stage():
        torch.distributed.send(output_tensor, get_pipeline_model_parallel_next_rank())


def send_backward(input_tensor_grad):
    if not is_pipeline_first_stage():
        torch.distributed.send(input_tensor_grad, get_pipeline_model_parallel_prev_rank())


def recv_forward(shape, dtype=torch.float32):
    input_tensor = None
    if not is_pipeline_first_stage():
        input_tensor = torch.empty(shape, requires_grad=True, device=torch.cuda.current_device(), dtype=dtype)

    torch.distributed.recv(input_tensor, get_pipeline_model_parallel_prev_rank())
    return input_tensor


def recv_backward(shape, dtype=torch.float32):
    output_tensor_grad = None
    if not is_pipeline_last_stage():
        output_tensor_grad = torch.empty(shape, requires_grad=True, device=torch.cuda.current_device(), dtype=dtype)

    torch.distributed.recv(output_tensor_grad, get_pipeline_model_parallel_next_rank())
    return output_tensor_grad


def send_forward_recv_backward(output_tensor, dtype=torch.float32):
    output_tensor_grad = None
    if not is_pipeline_last_stage():
        output_tensor_grad = torch.empty_like(output_tensor, requires_grad=True,
                                              device=torch.cuda.current_device(), dtype=dtype)
        send_op = torch.distributed.P2POp(torch.distributed.isend, output_tensor,
                                          get_pipeline_model_parallel_next_rank())
        recv_op = torch.distributed.P2POp(torch.distributed.irecv, output_tensor_grad,
                                          get_pipeline_model_parallel_next_rank())
        reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()

        torch.cuda.synchronize()
    else:
        torch.distributed.recv(output_tensor_grad, get_pipeline_model_parallel_next_rank())

    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad, dtype=torch.float32):
    input_tensor = None
    if not is_pipeline_first_stage():
        input_tensor = torch.empty_like(input_tensor_grad, requires_grad=True,
                                        device=torch.cuda.current_device(), dtype=dtype)
        send_op = torch.distributed.P2POp(torch.distributed.isend, input_tensor_grad,
                                          get_pipeline_model_parallel_prev_rank())
        recv_op = torch.distributed.P2POp(torch.distributed.irecv, input_tensor,
                                          get_pipeline_model_parallel_prev_rank())
        reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()

        torch.cuda.synchronize()
    else:
        torch.distributed.recv(input_tensor, get_pipeline_model_parallel_next_rank())

    return input_tensor


def pipedream_flush_schedule(data_iterator, model, loss_func):
    num_microbatches = get_num_microbatches()
    num_warmup_microbatches = get_pipeline_model_parallel_world_size() - \
        get_pipeline_model_parallel_rank() - 1
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    input_tensors = []
    output_tensors = []
    loss = torch.tensor(0.0)

    # run warmup forward passes
    for _ in range(num_warmup_microbatches):
        input_tensor = recv_forward(model.input_shape)
        output_tensor = forward_step(data_iterator, model, input_tensor, loss_func, loss)
        send_forward(output_tensor)

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(model.input_shape)

    # run 1F1B steady state
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))
        output_tensor = forward_step(data_iterator, model, input_tensor, loss_func, loss)
        output_tensor_grad = send_forward_recv_backward(output_tensor)

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)

        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

        if last_iteration:
            send_backward(input_tensor_grad)
        else:
            input_tensor = send_backward_recv_forward(input_tensor_grad)

    # run cooldown backward pass
    for i in range(num_warmup_microbatches):
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)

        output_tensor_grad = recv_backward(model.output_shape)

        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

        send_backward(input_tensor_grad)

    return loss.item()
