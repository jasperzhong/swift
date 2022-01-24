import math
import torch
from torch import optim
from torchvision.models import resnet18

model = resnet18(pretrained=True)	# 加载模型
optimizer = torch.optim.Adam(params=[	# 初始化优化器，并设置两个param_groups
    {'params': model.layer2.parameters()},
    {'params': model.layer3.parameters(), 'lr':0.2},
], lr=0.1)	# base_lr = 0.1

# 设置warm up的轮次为100次
warm_up_iter = 10
T_max = 50	# 周期
lr_max = 0.1	# 最大值
lr_min = 1e-5	# 最小值

# 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
lambda0 = lambda iter: iter / warm_up_iter if iter <= warm_up_iter \
                            else 0.5 * ( math.cos((iter - warm_up_iter) /(T_max - warm_up_iter) * math.pi) + 1)

#  param_groups[1] 不进行调整
lambda1 = lambda cur_iter: 1

# LambdaLR
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda0, lambda1])

num = 0
flag = True
for epoch in range(50):
    print(num, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
    optimizer.step()
    scheduler.step()
    num += 1

    if num == 10 and flag:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[scheduler.lr_lambdas[0], lambda1], last_epoch=7)
        scheduler.step()
        print("undo: {} {}".format(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
        optimizer.undo()
        num -= 1
        flag = False