# coding=utf-8
from __future__ import absolute_import, division, print_function
from benchmarks.distributed.ft.finetune_vit.schedule import is_pipeline_first_stage, is_pipeline_last_stage, get_num_microbatches, \
                                                            recv_forward, send_forward
import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

import torch
import time
import torch.nn as nn
from torchvision import datasets, transforms



logger = logging.getLogger(__name__)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def simple_accuracy(preds, labels):
    print("preds:{}".format(preds))
    print("labels:{}".format(labels))
    return (preds == labels).mean()

def fault_tolerance_val(config, model, test_loader, loss_func):
    # Validation!
    eval_losses = AverageMeter()
    accu = AverageMeter()

    test_iters = len(test_loader)
    print("test iters:{}".format(test_iters))
    logger.info("***** Running Validation *****")

    model.eval()
    all_preds, all_label = [], []
    
    data_iter = iter(test_loader)

    for i in range(test_iters):
        with torch.no_grad():
            if is_pipeline_last_stage():
                output_tensor, loss, labels = forward(config, data_iter, model, loss_func)
                eval_losses.update(loss, config.test_batch_size)

                top1 = compute_accuracy(output_tensor.detach(), labels)

                accu.update(top1, config.test_batch_sizes)
                # preds = torch.argmax(output_tensor, dim=-1)
                # if len(all_preds) == 0:
                #     all_preds.append(preds.detach().cpu().numpy())
                #     all_label.append(labels.detach().cpu().numpy())
                # else:
                #     all_preds[0] = np.append(
                #         all_preds[0], preds.detach().cpu().numpy(), axis=0
                #     )
                #     all_label[0] = np.append(
                #         all_label[0], labels.detach().cpu().numpy(), axis=0
                #     )
            else:
                loss, output_tensor = forward(config, data_iter, model, loss_func)
    if is_pipeline_last_stage():
        # all_preds, all_label = all_preds[0], all_label[0]
        # accuracy = simple_accuracy(all_preds, all_label)

        logger.info("\n")
        logger.info("Validation Results")
        logger.info("Valid Loss: %2.5f" % eval_losses.avg)
        logger.info("Valid Accuracy: %2.5f" % accu.avg)

        return accu.avg

def forward(config, data_iterator, model, loss_func):
    shape = (config.test_batch_size, *model.input_shape[1:])
    loss = 0
    input_tensor = recv_forward(shape)
    if is_pipeline_last_stage():
        output_tensor, loss, labels = forward_step(data_iterator, model, input_tensor, loss_func, loss)
        return output_tensor, loss, labels
    else:
        output_tensor = forward_step(data_iterator, model, input_tensor, loss_func, loss)
    send_forward(output_tensor)
    return loss, output_tensor
    
def get_transform_func():
    transform = nn.Sequential(
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        # ToTensor(transforms.ToTensor()),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )
    return transform

def forward_step(data_iterator, model, input_tensor, loss_func, loss):
    transforms = get_transform_func()
    if is_pipeline_first_stage() or is_pipeline_last_stage():
        data = next(data_iterator)
        images, labels = data

        if is_pipeline_first_stage():
            images = images.cuda()
            images = transforms(images)
        elif is_pipeline_last_stage():
            labels = labels.cuda()

    if is_pipeline_first_stage():
        assert input_tensor is None
        input_tensor = images

    output_tensor = model(input_tensor)

    if is_pipeline_last_stage():
        preds = output_tensor
        output_tensor = loss_func(output_tensor, labels)
        loss += output_tensor.item()
        
        return preds, loss, labels

    return output_tensor