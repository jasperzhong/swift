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

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def fault_tolerance_val(config, epoch, model, test_loader, loss_func):
    # Validation!
    eval_losses = AverageMeter()

    test_iters = config.test_iters
    logger.info("***** Running Validation *****")
    logger.info(" epoch {}".format(epoch) )

    model.eval()
    all_preds, all_label = [], []
    
    data_iter = iter(test_loader)
    labels = None
    for _ in range(test_iters):
        with torch.no_grad():
            loss, output_tensor = forward(data_iter, model, loss_func, eval_losses, all_preds, all_label)
    if is_pipeline_last_stage():
        all_preds, all_label = all_preds[0], all_label[0]
        accuracy = simple_accuracy(all_preds, all_label)

        logger.info("\n")
        logger.info("Validation Results")
        logger.info("Valid Loss: %2.5f" % eval_losses.avg)
        logger.info("Valid Accuracy: %2.5f" % accuracy)

        return accuracy

def forward(data_iterator, model, loss_func, eval_losses, all_preds, all_label):
    loss = 0
    input_tensor = recv_forward(model.input_shape)
    output_tensor = forward_step(data_iterator, model, input_tensor, loss_func, loss, eval_losses, all_preds, all_label)
    send_forward(output_tensor)
    print("forward:{}".foramt(all_preds))
    return loss, output_tensor
    
def get_transform_func():
    transform = nn.Sequential(
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        # ToTensor(transforms.ToTensor()),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )
    return transform

def forward_step(data_iterator, model, input_tensor, loss_func, loss, eval_losses, all_preds, all_label):
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
        output_tensor = loss_func(output_tensor, labels)
        loss += output_tensor.item()
        eval_losses.update(loss)
        preds = torch.argmax(output_tensor, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(labels.detach().cpu().numpy())
            print(all_preds)
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], labels.detach().cpu().numpy(), axis=0
            )

    return output_tensor