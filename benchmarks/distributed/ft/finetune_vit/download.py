import torch
from torchvision import datasets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

transform_train = transforms.Compose([
        transforms.RandomResizedCrop((384, 384), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

trainset = datasets.CIFAR100(root="~/data/",
                            train=True,
                            download=True,
                            transform=transform_train)

train_sampler = RandomSampler(trainset)
train_loader = DataLoader(trainset,
                            sampler=train_sampler,
                            batch_size=1,
                            num_workers=4,
                            pin_memory=True)
train_iter = iter(train_loader)

for i in range(1):
    tensor = next(train_iter)
    print(tensor[0].shape)
