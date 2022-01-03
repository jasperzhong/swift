import copy
import itertools
import unittest
import time

import numpy as np
from parameterized import parameterized
from torchvision.models import resnet50

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim._functional as F


def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )

def checksum(model, optimizer):
    model_sum = 0
    for param in model.parameters():
        model_sum += torch.sum(param)

    optimizer_sum = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                state = optimizer.state[p]
                if 'momentum_buffer' in state:
                    optimizer_sum += torch.sum(state['momentum_buffer'])
                if 'step' in state:
                    optimizer_sum += state['step']
                if 'exp_avg' in state:
                    optimizer_sum += torch.sum(state['exp_avg'])
                if 'exp_avg_sq' in state:
                    optimizer_sum += torch.sum(state['exp_avg_sq'])
                

    return model_sum, optimizer_sum

def checksum(model, optimizer):
    model_sum = 0
    for param in model.parameters():
        model_sum += torch.sum(param)

    optimizer_sum = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                state = optimizer.state[p]
                if 'momentum_buffer' in state:
                    optimizer_sum += torch.sum(state['momentum_buffer'])

    return model_sum, optimizer_sum


class UndoTestCase(unittest.TestCase):
    # @parameterized.expand(itertools.product([0, 1e-4], [0, 0.9], [False, True]), name_func=custom_name_func)
    # def test_undo_sgd(self, wd, momentum, nesterov):
    #     params = []
    #     params_copy = []
    #     d_p_list = []
    #     momentum_buffer_list = []
    #     momentum_buffer_list_copy = []

    #     num_tensors = 50 
    #     size = (32, 32)
    #     for _ in range(num_tensors):
    #         param = torch.randn(size=size, requires_grad=True).cuda()
    #         param_copy = param.clone()
    #         params.append(param)
    #         params_copy.append(param_copy)
    #         d_p_list.append(torch.randn(size=size).cuda())
    #         momentum_buffer = torch.randn(size=size).cuda()
    #         momentum_buffer_copy = momentum_buffer.clone()
    #         momentum_buffer_list.append(momentum_buffer)
    #         momentum_buffer_list_copy.append(momentum_buffer_copy)

    #     lr = 0.1
    #     with torch.no_grad():
    #         F.sgd(params, d_p_list, momentum_buffer_list, weight_decay=wd,
    #               momentum=momentum, lr=lr, dampening=0, nesterov=nesterov)

    #         F.undo_sgd(params, d_p_list, momentum_buffer_list, weight_decay=wd,
    #                    momentum=momentum, lr=lr, dampening=0, nesterov=nesterov)

    #     atol = torch.finfo(torch.float).resolution
    #     for i in range(num_tensors):
    #         if torch.allclose(params[i], params_copy[i], atol=atol) is False:
    #             print(params[i] - params_copy[i])
    #             self.assertTrue(False)

    #         if torch.allclose(momentum_buffer_list[i], momentum_buffer_list_copy[i], atol=atol) is False:
    #             print(momentum_buffer_list[i] - momentum_buffer_list_copy[i])
    #             self.assertTrue(False)

    # @parameterized.expand(itertools.product([0.9, 0.95], [0.999, 0.999], [0, 1e-4], [1e-8, 1e-9]), name_func=custom_name_func)
    # def test_undo_adam(self, b1, b2, wd, eps):
    #     params = []
    #     params_copy = []
    #     grads = []
    #     exp_avg_sqs = []
    #     exp_avg_sqs_copy = []
    #     exp_avgs = []
    #     exp_avgs_copy = []
    #     state_steps = []

    #     num_tensors = 50 
    #     size = (32, 32)
    #     for _ in range(num_tensors):
    #         param = torch.randn(size=size, requires_grad=True).cuda()
    #         param_copy = param.clone()
    #         params.append(param)
    #         params_copy.append(param_copy)
    #         grads.append(torch.randn(size=size).cuda())
    #         exp_avg = torch.randn_like(param, memory_format=torch.preserve_format).abs().cuda()
    #         exp_avgs.append(exp_avg)
    #         exp_avg_copy = exp_avg.clone()
    #         exp_avgs_copy.append(exp_avg_copy)
    #         exp_avg_sq = torch.randn_like(param, memory_format=torch.preserve_format).abs().cuda()
    #         exp_avg_sqs.append(exp_avg_sq)
    #         exp_avg_sq_copy = exp_avg_sq.clone()
    #         exp_avg_sqs_copy.append(exp_avg_sq_copy)
    #         state_steps.append(1)

    #     lr = 0.1
    #     with torch.no_grad():
    #         F.adam(params, grads, exp_avgs=exp_avgs, exp_avg_sqs=exp_avg_sqs, max_exp_avg_sqs=[], state_steps=state_steps, amsgrad=False, 
    #                 beta1=b1, beta2=b2, lr=lr, weight_decay=wd, eps=eps)

    #         F.undo_adam(params, grads, exp_avgs=exp_avgs, exp_avg_sqs=exp_avg_sqs, state_steps=state_steps, 
    #                 beta1=b1, beta2=b2, lr=lr, weight_decay=wd, eps=eps)

    #     atol = torch.finfo(torch.float).resolution
    #     for i in range(num_tensors):
    #         if torch.allclose(params[i], params_copy[i], atol=atol) is False:
    #             print(params[i] - params_copy[i])
    #             self.assertTrue(False)

    #         if torch.allclose(exp_avgs[i], exp_avgs_copy[i], atol=atol) is False:
    #             print(exp_avgs[i] - exp_avgs_copy[i])
    #             self.assertTrue(False)

    #         if torch.allclose(exp_avg_sqs[i], exp_avg_sqs_copy[i], atol=atol) is False:
    #             print(exp_avg_sqs[i] - exp_avg_sqs_copy[i])
    #             self.assertTrue(False)

    # @parameterized.expand(itertools.product([0.9, 0.95], [0.999, 0.999], [0, 1e-4], [1e-8, 1e-9]), name_func=custom_name_func)
    # def test_undo_adamw(self, b1, b2, wd, eps):
    #     params = []
    #     params_copy = []
    #     grads = []
    #     exp_avg_sqs = []
    #     exp_avg_sqs_copy = []
    #     exp_avgs = []
    #     exp_avgs_copy = []
    #     state_steps = []

    #     num_tensors = 50 
    #     size = (32, 32)
    #     for _ in range(num_tensors):
    #         param = torch.randn(size=size, requires_grad=True).cuda()
    #         param_copy = param.clone()
    #         params.append(param)
    #         params_copy.append(param_copy)
    #         grads.append(torch.randn(size=size).cuda())
    #         exp_avg = torch.ones_like(param, memory_format=torch.preserve_format).cuda()
    #         exp_avgs.append(exp_avg)
    #         exp_avg_copy = exp_avg.clone()
    #         exp_avgs_copy.append(exp_avg_copy)
    #         exp_avg_sq = torch.ones_like(param, memory_format=torch.preserve_format).cuda()
    #         exp_avg_sqs.append(exp_avg_sq)
    #         exp_avg_sq_copy = exp_avg_sq.clone()
    #         exp_avg_sqs_copy.append(exp_avg_sq_copy)
    #         state_steps.append(1)

    #     lr = 0.1
    #     with torch.no_grad():
    #         F.adamw(params, grads, exp_avgs=exp_avgs, exp_avg_sqs=exp_avg_sqs, max_exp_avg_sqs=[], state_steps=state_steps, amsgrad=False, 
    #                 beta1=b1, beta2=b2, lr=lr, weight_decay=wd, eps=eps)

    #         F.undo_adamw(params, grads, exp_avgs=exp_avgs, exp_avg_sqs=exp_avg_sqs, state_steps=state_steps, 
    #                 beta1=b1, beta2=b2, lr=lr, weight_decay=wd, eps=eps)

    #     atol = torch.finfo(torch.float).resolution
    #     for i in range(num_tensors):
    #         if torch.allclose(params[i], params_copy[i], atol=atol) is False:
    #             print(params[i] - params_copy[i])
    #             self.assertTrue(False)

    #         if torch.allclose(exp_avgs[i], exp_avgs_copy[i], atol=atol) is False:
    #             print(exp_avgs[i] - exp_avgs_copy[i])
    #             self.assertTrue(False)

    #         if torch.allclose(exp_avg_sqs[i], exp_avg_sqs_copy[i], atol=atol) is False:
    #             print(exp_avg_sqs[i] - exp_avg_sqs_copy[i])
    #             self.assertTrue(False)
                
    def test_undo_adam_train_resnet50(self):
        model = resnet50().cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        loss_func = nn.CrossEntropyLoss().cuda()

        # 设置warm up的轮次为100次
        warm_up_iter = 10
        T_max = 50  # 周期
        lr_max = 0.1  # 最大值
        lr_min = 1e-5  # 最小值

        # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
        def lambda0(iter): return iter / warm_up_iter if iter <= warm_up_iter \
            else 0.5 * (math.cos((iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

        num = 0
        size = (32, 3, 224, 224)
        x = torch.randn(size=size).cuda()
        y = torch.randint(0, 1000, (32, )).cuda()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_func(y_hat, y)
        loss.backward()
        print(num, optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
        num += 1

        model_sum_1, optimizer_sum_1 = checksum(model, optimizer)

        x = torch.randn(size=size).cuda()
        y = torch.randint(0, 1000, (32, )).cuda()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_func(y_hat, y)
        loss.backward()
        print(num, optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
        num += 1

        num -= 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler.lr_lambdas[0], last_epoch=num - 1)

        print("undo: {}".format(optimizer.param_groups[0]['lr']))
        optimizer.undo()
        model_sum_2, optimizer_sum_2 = checksum(model, optimizer)

        print("model sum diff = {:.6f}".format(torch.abs(model_sum_1 - model_sum_2)))
        print("optimizer sum diff = {:.6f}".format(torch.abs(optimizer_sum_1 - optimizer_sum_2)))
        print("{:.6f} {:.6f}".format(optimizer_sum_1, optimizer_sum_2))
        
    def test_undo_adamw_train_resnet50(self):
        model = resnet50().cuda()
        optimizer = optim.AdamW(model.parameters(), lr=0.1)
        loss_func = nn.CrossEntropyLoss().cuda()

        # 设置warm up的轮次为100次
        warm_up_iter = 10
        T_max = 50  # 周期
        lr_max = 0.1  # 最大值
        lr_min = 1e-5  # 最小值

        # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
        def lambda0(iter): return iter / warm_up_iter if iter <= warm_up_iter \
            else 0.5 * (math.cos((iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

        num = 0
        size = (32, 3, 224, 224)
        x = torch.randn(size=size).cuda()
        y = torch.randint(0, 1000, (32, )).cuda()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_func(y_hat, y)
        loss.backward()
        print(num, optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
        num += 1
        model_sum_1, optimizer_sum_1 = checksum(model, optimizer)

        x = torch.randn(size=size).cuda()
        y = torch.randint(0, 1000, (32, )).cuda()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_func(y_hat, y)
        loss.backward()
        print(num, optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
        num += 1

        num -= 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler.lr_lambdas[0], last_epoch=num - 1)

        print("undo: {}".format(optimizer.param_groups[0]['lr']))
        optimizer.undo()
        model_sum_2, optimizer_sum_2 = checksum(model, optimizer)

        print("model sum diff = {:.6f}".format(torch.abs(model_sum_1 - model_sum_2)))
        print("optimizer sum diff = {:.6f}".format(torch.abs(optimizer_sum_1 - optimizer_sum_2)))
        print("{:.6f} {:.6f}".format(optimizer_sum_1, optimizer_sum_2))

if __name__ == "__main__":
    unittest.main()
