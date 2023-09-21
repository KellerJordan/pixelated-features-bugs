import os

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.cuda.amp import autocast, GradScaler

from loader import CifarLoader
from models import construct_rn9, construct_rn18, construct_speedyrn, construct_vgg11

@autocast()
@torch.no_grad()
def evaluate(model):
    model.eval()
    correct = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        correct += (outputs.argmax(1) == labels).sum().item()
    return correct

def train(construct_fn, epochs, lr, name):
    """
    Hardcoded options:
    * warmup_epochs=5
    * momentum=0.9
    * weight_decay=5e-4
    """
    n_iters = epochs*len(train_loader)
    lr_schedule = np.interp(np.arange(1+n_iters), [0, 5*len(train_loader), n_iters], [0, 1, 0])

    model = construct_fn(pixelate_option)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    scaler = GradScaler()

    losses = []
    for epoch in range(epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels)
            losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        print(name, epoch, evaluate(model))

    torch.save(model.state_dict(), os.path.join(save_dir, f'model_{name}.pt'))

if __name__ == '__main__':
    pixelate_option = (2, 4)

    save_dir = 'cifar_nocutout_%d_%d_v1' % pixelate_option
    os.makedirs(save_dir, exist_ok=True)

    bs = 500
    train_augs = dict(flip=True, translate=2, cutout=8, pixelate=pixelate_option)
    train_loader = CifarLoader('cifar10/', train=True, batch_size=bs, aug=train_augs)
    test_augs = dict(pixelate=pixelate_option)
    test_loader = CifarLoader('cifar10/', train=False, batch_size=2*bs, aug=test_augs)

    train(construct_rn9, 32, 0.5, 'resnet9')
    train(construct_speedyrn, 24, 0.5, 'speedyresnet')
    train(construct_vgg11, 32, 0.05, 'vgg11')
    train(construct_rn18, 24, 0.5, 'resnet18')

