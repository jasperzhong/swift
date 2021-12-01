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



if __name__ == "__main__":
    unittest.main()
