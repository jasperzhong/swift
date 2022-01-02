from torchvision import datasets

trainset = datasets.CIFAR100(root="~/data/CIFAR100",
                             train=True,
                             download=True)
testset = datasets.CIFAR100(root="~/data/CIFAR100",
                            train=False,
                            download=True)
