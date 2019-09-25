"""
Benchmark to show performance of individual gradient computations

The procedures we test are:

- Compute all individual gradients using
    - BackPACK
    - a for-loop
- Compute the variance using
    - a for-loop to compute the individual gradients
    - BackPACK to compute the individual gradients
    - a for-loop to compute the first and second moments
    - BackPACK to compute the first and second moments

on a small (Logreg MNIST) and large (CIFAR10 3C3D) dataset
using different batch sizes.
"""
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from backpack import backpack, extend
from benchmark_networks import net_cifar100_allcnnc, net_cifar10_3c3d, net_fmnist_2c2d

"""
Data loading
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cifar_transform = transforms.Compose([
    transforms.Pad(padding=2),
    transforms.RandomCrop(size=(32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=63. / 255.,
                           saturation=[0.5, 1.5],
                           contrast=[0.2, 1.8]),
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                         (0.24703223, 0.24348513, 0.26158784))
])


def make_loader_for_dataset(dataset):
    def loader(batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

    return loader


def data_prep_cifar100(use_sigmoid=False):
    model = extend(net_cifar100_allcnnc(use_sigmoid)).to(device)
    lossfunc = extend(nn.CrossEntropyLoss())

    dataset = datasets.CIFAR100(
        './data',
        train=True,
        download=True,
        transform=cifar_transform
    )

    return model, lossfunc, make_loader_for_dataset(dataset)


def data_prep_cifar10(use_sigmoid=False):
    model = extend(net_cifar10_3c3d(use_sigmoid)).to(device)
    lossfunc = extend(nn.CrossEntropyLoss())

    dataset = datasets.CIFAR10(
        './data',
        train=True,
        download=True,
        transform=cifar_transform
    )

    return model, lossfunc, make_loader_for_dataset(dataset)


def data_prep_fmnist_sigmoid():
    model = extend(()).to(device)
    lossfunc = extend(nn.CrossEntropyLoss())

    dataset = datasets.FashionMNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    def make_loader(batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

    return model, lossfunc, make_loader


def get_first_batch(loader):
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        return data, target


"""
Functions to test
"""


def make_backpack_extension_runner(loader, model, lossfunc, *extensions):
    def backpack_extension_run():
        X, Y = get_first_batch(loader)
        loss = lossfunc(model(X), Y)

        with backpack(*extensions):
            loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=None)

        return [[getattr(p, ext.savefield) for p in model.parameters()] for ext in extensions]

    return backpack_extension_run


def makefunc_batchgrad_forloop(loader, model, lossfunc):
    def forloop_batchgrad():
        X, Y = get_first_batch(loader)
        grads = []
        for n in range(X.shape[0]):
            xn = X[n, :].unsqueeze(0)
            yn = Y[n].unsqueeze(0)
            loss = lossfunc(model(xn), yn)
            grads.append(torch.autograd.grad(loss, model.parameters()))

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=None)

        return grads

    return forloop_batchgrad


def makefunc_grad(loader, model, lossfunc):
    def grad():
        X, Y = get_first_batch(loader)
        loss = lossfunc(model(X), Y)
        loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=None)

        return [p.grad for p in model.parameters()]

    return grad
