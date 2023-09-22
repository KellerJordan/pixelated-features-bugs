"""
Analyzes generated adversarial perturbations for transferability.
"""

import os
import torch

if __name__ == '__main__':
    pixelate_option = (2, 5)
    save_dir = 'cifar_%d_%d_v5' % pixelate_option
    obj = torch.load(os.path.join(save_dir, 'perturbations.pt'))

    kk = ['perturbations', 'perturbations_feature', 'perturbations_bug']
    for k in kk:
        uu = obj[k]
        uu1 = uu.clone().permute(1, 0, 2)
        uu1 /= uu1.norm(dim=2, keepdim=True)
        dots = torch.stack([u @ u.T for u in uu1])
        dots_mu = dots.mean(0)
        print(k)
        print('average norms')
        avg_norm = uu.norm(dim=2).square().mean(1).sqrt()
        print(avg_norm)
        print('transferability')
        print(dots_mu)

