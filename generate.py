"""
Generates single-step adversarial perturbations for all (model, test example) pairs.
"""

import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from torchvision import transforms

from loader import CifarLoader, batch_pixelate
from models import construct_rn9, construct_rn18, construct_speedyrn, construct_vgg11


loader = CifarLoader('cifar10/', train=False, batch_size=1, aug=None)
normalize = transforms.Normalize(loader.mean, loader.std)
denormalize = transforms.Normalize(-loader.mean/loader.std, 1/loader.std)
def get_grad(model, img, tgt, pixelate_option):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    delta = torch.zeros_like(img)
    delta.requires_grad = True
    input1 = normalize(denormalize(img + delta).clip(0, 1))
    out = model(input1)
    tgt_logit = out[:, tgt]
    other_logits = torch.cat([out[:, :tgt], out[:, tgt+1:]], dim=1)
    margin = tgt_logit - other_logits.amax(dim=1)
    margin.backward()
    gg = -delta.grad.flatten()
    
    # compute gp as projection of gg onto pixelated subspace
    gp = batch_pixelate(gg.reshape(1, 3, img.size(2), img.size(2)),
                        pixelate_option[1], pixelate_option[1]).flatten().cpu()
    gg = gg.cpu()
    # compute gf as remaining high-frequency component of gradient
    gf = gg - (gg @ gp) * gp / gp.norm()**2
    
    return (gg, gp, gf)

if __name__ == '__main__':
    pixelate_option = (2, 5)
    save_dir = 'cifar_%d_%d_v5' % pixelate_option

    model1 = construct_rn9(pixelate_option)
    model1.load_state_dict(torch.load(save_dir+'/model_resnet9.pt'))
    model2 = construct_speedyrn(pixelate_option)
    model2.load_state_dict(torch.load(save_dir+'/model_speedyresnet.pt'))
    model3 = construct_vgg11(pixelate_option)
    model3.load_state_dict(torch.load(save_dir+'/model_vgg11.pt'))
    model4 = construct_rn18(pixelate_option)
    model4.load_state_dict(torch.load(save_dir+'/model_resnet18.pt'))
    models = [model1, model2, model3, model4]

    test_augs = dict(pixelate=pixelate_option)
    test_loader = CifarLoader('cifar10/', train=False, batch_size=500, aug=test_augs, shuffle=False)
    inputs = test_loader.images.float()
    labels = test_loader.targets

    uus_list = []
    for x0, y0 in tqdm(zip(inputs, labels), total=len(inputs)):
        uus = [get_grad(model, x0[None], y0.item(), pixelate_option)
               for model in models]
        uus_list.append(uus)

    by_model = list(zip(*uus_list))
    by_model1 = [list(map(torch.stack, zip(*row))) for row in by_model]
    uu, up, uf = [torch.stack(x) for x in zip(*by_model1)]

    obj = {
        'perturbations': uu,
        'perturbations_feature': up,
        'perturbations_bug': uf,
    }
    obj['images'] = test_loader.images.cpu()
    obj['labels'] = test_loader.targets.cpu()
    obj['models'] = ['resnet9', 'speedyrn', 'vgg11', 'resnet18']

    torch.save(obj, os.path.join(save_dir, 'perturbations.pt'))

