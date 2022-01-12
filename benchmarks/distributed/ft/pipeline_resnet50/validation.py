import torch
import logging
import schedule
from enum import Enum
from schedule import recv_forward, send_forward, is_pipeline_last_stage, is_pipeline_first_stage

logger = logging.getLogger(__name__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

def fault_tolerance_val(config, model, test_loader, loss_func):
    # Validation!
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    test_iters = len(test_loader)
    print("test iters:{}".format(test_iters))
    logger.info("***** Running Validation *****")

    model.eval()

    data_iter = iter(test_loader)

    for i in range(test_iters):
        with torch.no_grad():
            if is_pipeline_last_stage():
                output_tensor, loss, labels = forward(config, data_iter, model, loss_func)
                losses.update(loss, schedule._GLOBAL_ARGS.test_batch_size)

                acc1, acc5 = accuracy(output_tensor, labels, topk=(1, 5))

                top1.update(acc1, schedule._GLOBAL_ARGS.test_batch_size)
                # top5.update(acc5, schedule._GLOBAL_ARGS.test_batch_size)

            else:
                loss, output_tensor = forward(config, data_iter, model, loss_func)
    if is_pipeline_last_stage():

        logger.info("\n")
        logger.info("Validation Results")
        logger.info("Valid Loss: %2.5f" % losses.avg)
        logger.info("Valid Accuracy: %2.5f" % top1.avg)

        return top1.avg
    
    return 0


def forward(config, data_iterator, model, loss_func):
    shape = (schedule._GLOBAL_ARGS.test_batch_size, *model.input_shape[1:])
    loss = 0
    input_tensor = recv_forward(shape)
    if is_pipeline_last_stage():
        output_tensor, loss, labels = forward_step(data_iterator, model, input_tensor, loss_func, loss)
        return output_tensor, loss, labels
    else:
        output_tensor = forward_step(data_iterator, model, input_tensor, loss_func, loss)
    send_forward(output_tensor)
    return loss, output_tensor

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
        preds = output_tensor
        print(f"labels shape: {labels.shape}")
        output_tensor = loss_func(output_tensor, labels)
        loss += output_tensor.item()

        return preds, loss, labels

    return output_tensor