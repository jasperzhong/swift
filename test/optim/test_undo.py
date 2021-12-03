from torch._C import memory_format
import torch
import torch.optim._functional as F
import unittest
from parameterized import parameterized
import itertools


def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )


class UndoTestCase(unittest.TestCase):
    @parameterized.expand(itertools.product([0, 1e-4], [0, 0.9], [False, True]), name_func=custom_name_func)
    def test_undo_sgd(self, wd, momentum, nesterov):
        params = []
        params_copy = []
        d_p_list = []
        momentum_buffer_list = []
        momentum_buffer_list_copy = []

        num_tensors = 50 
        size = (32, 32)
        for _ in range(num_tensors):
            param = torch.randn(size=size, requires_grad=True).cuda()
            param_copy = param.clone()
            params.append(param)
            params_copy.append(param_copy)
            d_p_list.append(torch.randn(size=size).cuda())
            momentum_buffer = torch.randn(size=size).cuda()
            momentum_buffer_copy = momentum_buffer.clone()
            momentum_buffer_list.append(momentum_buffer)
            momentum_buffer_list_copy.append(momentum_buffer_copy)

        lr = 0.1
        with torch.no_grad():
            F.sgd(params, d_p_list, momentum_buffer_list, weight_decay=wd,
                  momentum=momentum, lr=lr, dampening=0, nesterov=nesterov)

            F.undo_sgd(params, d_p_list, momentum_buffer_list, weight_decay=wd,
                       momentum=momentum, lr=lr, dampening=0, nesterov=nesterov)

        atol = torch.finfo(torch.float).resolution
        for i in range(num_tensors):
            if torch.allclose(params[i], params_copy[i], atol=atol) is False:
                print(params[i] - params_copy[i])
                self.assertTrue(False)

            if torch.allclose(momentum_buffer_list[i], momentum_buffer_list_copy[i], atol=atol) is False:
                print(momentum_buffer_list[i] - momentum_buffer_list_copy[i])
                self.assertTrue(False)

    @parameterized.expand(itertools.product([0.9, 0.95], [0.999, 0.999], [0, 1e-4], [1e-8, 1e-9]), name_func=custom_name_func)
    def test_undo_adam(self, b1, b2, wd, eps):
        params = []
        params_copy = []
        grads = []
        exp_avg_sqs = []
        exp_avg_sqs_copy = []
        exp_avgs = []
        exp_avgs_copy = []
        state_steps = []

        num_tensors = 50 
        size = (32, 32)
        for _ in range(num_tensors):
            param = torch.randn(size=size, requires_grad=True).cuda()
            param_copy = param.clone()
            params.append(param)
            params_copy.append(param_copy)
            grads.append(torch.randn(size=size).cuda())
            exp_avgs = torch.randn_like(param, memory_format=torch.preserve_format).cuda()
            exp_avgs_copy = exp_avgs.clone()
            exp_avg_sqs = torch.randn_like(param, memory_format=torch.preserve_format).cuda()
            exp_avg_sqs_copy = exp_avg_sqs.clone()
            state_steps = torch.ones(size=size).cuda()

        lr = 0.1
        with torch.no_grad():
            F.adam(params, grads, exp_avgs=exp_avgs, exp_avg_sqs=exp_avg_sqs, state_steps=state_steps, amsgrad=False, 
                    beta1=b1, beta2=b2, lr=lr, weight_decay=wd, eps=eps)

            F.undo_adam(params, grads, exp_avgs=exp_avgs, exp_avg_sqs=exp_avg_sqs, state_steps=state_steps, amsgrad=False, 
                    beta1=b1, beta2=b2, lr=lr, weight_decay=wd, eps=eps)

        atol = torch.finfo(torch.float).resolution
        for i in range(num_tensors):
            if torch.allclose(params[i], params_copy[i], atol=atol) is False:
                print(params[i] - params_copy[i])
                self.assertTrue(False)

            if torch.allclose(exp_avgs[i], exp_avgs_copy[i], atol=atol) is False:
                print(exp_avgs[i] - exp_avgs_copy[i])
                self.assertTrue(False)

            if torch.allclose(exp_avg_sqs[i], exp_avg_sqs_copy[i], atol=atol) is False:
                print(exp_avg_sqs[i] - exp_avg_sqs_copy[i])
                self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
