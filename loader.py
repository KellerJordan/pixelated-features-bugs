"""
Fast CIFAR-10 dataloader.
Modified from https://github.com/KellerJordan/cifar10-loader to support "pixelated" images.
"""
import os
from math import ceil
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py#L389
def make_random_square_masks(inputs, size, pixel=1):
    is_even = int(size % 2 == 0)
    n,c,h,w = inputs.shape

    # seed top-left corners of squares to cutout boxes from, in one dimension each
    corner_y = torch.randint(0, h-size+1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w-size+1, size=(n,), device=inputs.device)
    
    corner_y = pixel * (corner_y // pixel)
    corner_x = pixel * (corner_x // pixel)

    # measure distance, using the center as a reference point
    corner_y_dists = torch.arange(h, device=inputs.device).view(1, 1, h, 1) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1, 1, 1, w) - corner_x.view(-1, 1, 1, 1)
    
    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)

    final_mask = mask_y * mask_x

    return final_mask

def batch_flip_lr(batch_images, flip_chance=.5):
    return torch.where(torch.rand_like(batch_images[:, 0, 0, 0]).view(-1, 1, 1, 1) < flip_chance,
                       batch_images.flip(-1), batch_images)

def batch_crop(inputs, crop_size, pixel=1):
    crop_mask_batch = make_random_square_masks(inputs, crop_size, pixel)
    cropped_batch = torch.masked_select(inputs, crop_mask_batch)
    return cropped_batch.view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)

def batch_translate(inputs, translate, pixel=1):
    translate = pixel * (translate // pixel)
    width = inputs.shape[2]
    inputs = F.pad(inputs, (translate,)*4, 'constant', value=0)
    return batch_crop(inputs, width, pixel)

def batch_cutout(inputs, size):
    masks = make_random_square_masks(inputs, size)
    cutout_batch = inputs.masked_fill(masks, 0)
    return cutout_batch

# downscales and then upscales to a pixelated-looking image
def batch_pixelate(inputs, size1, size2=None):
    if size2 is None:
        size2 = size1
    pixelate = nn.Sequential(
        nn.Conv2d(3, 3, size1, stride=size1, padding=0, bias=False),
        nn.ConvTranspose2d(3, 3, size2, stride=size2, padding=0, bias=False),
    )
    pixelate[0].weight.data = 1/size1**2 * (torch.eye(3)[:, :, None, None] * torch.ones(1, 1, size1, size1))
    pixelate[1].weight.data = 1.0 * (torch.eye(3)[:, :, None, None] * torch.ones(1, 1, size2, size2))
    pixelate = pixelate.cuda().type(inputs.dtype)
    for p in pixelate.parameters():
        p.requires_grad = False
    return pixelate(inputs)

class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None, keep_last=False, shuffle=True, gpu=0):
        dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
        imgs = torch.tensor(dset.data, dtype=torch.half).cuda(gpu)
        imgs = (imgs / 255).permute(0, 3, 1, 2)
        imgs = imgs.to(memory_format=torch.channels_last)
        self.mean = torch.tensor(CIFAR_MEAN, dtype=torch.half).view(1, 3, 1, 1).cuda(gpu)
        self.std = torch.tensor(CIFAR_STD, dtype=torch.half).view(1, 3, 1, 1).cuda(gpu)
        self.images = (imgs - self.mean) / self.std
        self.targets = torch.tensor(dset.targets).cuda(gpu)
        
        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate', 'cutout', 'pixelate'], 'Unrecognized key: %s' % k

        if 'pixelate' in self.aug:
            self.images = batch_pixelate(self.images, *self.aug['pixelate'])

        self.batch_size = batch_size
        self.keep_last = keep_last
        self.shuffle = shuffle

    def augment(self, images):
        if self.aug.get('flip', False):
            images = batch_flip_lr(images)
        if self.aug.get('cutout', 0) > 0:
            images = batch_cutout(images, self.aug['cutout'])
        if self.aug.get('translate', 0) > 0:
            # apply translation in minibatches of 5000 in order to save memory
            bs = 5000
            nb = ceil(len(images)/bs)
            pixel = self.aug['pixelate'][1] if 'pixelate' in self.aug else 1
            images = torch.cat([batch_translate(images[i*bs:(i+1)*bs], self.aug['translate'], pixel)
                                for i in range(nb)])
        return images

    def __len__(self):
        return ceil(len(self.images)/self.batch_size) if self.keep_last else len(self.images)//self.batch_size

    def __iter__(self):
        images = self.augment(self.images)
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs].float(), self.targets[idxs])

