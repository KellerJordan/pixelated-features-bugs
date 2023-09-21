"""
Defines 4 network architectures for CIFAR-10 training:
* ResNet-18
* VGG-11
* ResNet-9 from https://docs.ffcv.io/ffcv_examples/cifar10.html
* SpeedyResnet from https://github.com/tysam-code/hlb-CIFAR10
"""

import torch
from torch import nn
import torch.nn.functional as F
from loader import CifarLoader

## ResNet-9
# https://docs.ffcv.io/ffcv_examples/cifar10.html
def construct_rn9(pixelate_option):
    w = 1.0

    class Mul(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
        def forward(self, x):
            return x * self.weight

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    class Residual(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, x):
            return x + self.module(x)

    def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
                nn.Conv2d(channels_in, channels_out,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=False),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True)
        )

    NUM_CLASSES = 10
    w0 = int(w*64)
    w1 = int(w*128)
    w2 = int(w*256)
    model = nn.Sequential(
        conv_bn(3, w0, kernel_size=3, stride=1, padding=1),
        conv_bn(w0, w1, kernel_size=5, stride=2, padding=2),
        Residual(nn.Sequential(conv_bn(w1, w1), conv_bn(w1, w1))),
        conv_bn(w1, w2, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        Residual(nn.Sequential(conv_bn(w2, w2), conv_bn(w2, w2))),
        conv_bn(w2, w1, kernel_size=3, stride=1, padding=0),
        nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        nn.Linear(w1, NUM_CLASSES, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=torch.channels_last)
    return model


## SpeedyResNet
# https://github.com/tysam-code/hlb-CIFAR10
default_conv_kwargs = {'kernel_size': 3, 'padding': 'same'}
class Conv(nn.Conv2d):
    def __init__(self, *args, norm=False, **kwargs):
        super().__init__(*args, **{**default_conv_kwargs, **kwargs})

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = False
        self.bias.requires_grad = True

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = Conv(channels_in, channels_out)
        self.conv2 = Conv(channels_out, channels_out)

        self.norm1 = BatchNorm(channels_out)
        self.norm2 = BatchNorm(channels_out)

        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.activ(x)
        residual = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = x + residual
        return x

class TemperatureScaler(nn.Module):
    def __init__(self, init_val):
        super().__init__()
        self.scale = torch.tensor(init_val)
    def forward(self, x):
        return self.scale * x

class FastGlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # requires less time than AdaptiveMax2dPooling -- about ~.3s for the entire run
        return torch.amax(x, dim=(2,3))
    

def get_patches(x, patch_shape=(3, 3), dtype=torch.float32):
    # This uses the unfold operation (https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html?highlight=unfold#torch.nn.functional.unfold)
    # to extract a _view_ (i.e., there's no data copied here) of blocks in the input tensor. We have to do it twice -- once horizontally, once vertically. Then
    # from that, we get our kernel_size*kernel_size patches to later calculate the statistics for the whitening tensor on
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).to(dtype)

def get_whitening_parameters(patches):
    # As a high-level summary, we're basically finding the high-dimensional oval that best fits the data here.
    # We can then later use this information to map the input information to a nicely distributed sphere, where also
    # the most significant features of the inputs each have their own axis. This significantly cleans things up for the
    # rest of the neural network and speeds up training.
    n,c,h,w = patches.shape
    est_covariance = torch.cov(patches.view(n, c*h*w).t())
    eigenvalues, eigenvectors = torch.linalg.eigh(est_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.t().reshape(c*h*w,c,h,w).flip(0)

@torch.no_grad()
def init_whitening_conv(layer, train_set, pad=None):
    if pad > 0:
        train_set = train_set[:,:,pad:-pad,pad:-pad]
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvalues = eigenvalues.type(layer.weight.dtype)
    eigenvectors = eigenvectors.type(layer.weight.dtype)
    
    eps = 1e-12
    layer.weight.data = (eigenvectors/torch.sqrt(eigenvalues+eps))
    layer.weight.requires_grad = False


depths = {
    'init':   32,
    'block1': 64,
    'block2': 256,
    'block3': 512,
}

class SpeedyResNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict

    def forward(self, x):
        x = self.net_dict['initial_block']['whiten'](x)
        x = self.net_dict['initial_block']['project'](x)
        x = self.net_dict['initial_block']['activation'](x)
        x = self.net_dict['residual1'](x)
        x = self.net_dict['residual2'](x)
        x = self.net_dict['residual3'](x)
        x = self.net_dict['pooling'](x)
        x = self.net_dict['linear'](x)
        x = self.net_dict['temperature'](x)
        return x

def construct_speedyrn(pixelate_option):
    w=1
    whitening_kernel_size = 2
    
    whiten_conv_depth = 3*whitening_kernel_size**2
    network_dict = nn.ModuleDict({
        'initial_block': nn.ModuleDict({
            'whiten': Conv(3, int(w*whiten_conv_depth), kernel_size=whitening_kernel_size, padding=0),
            'project': Conv(int(w*whiten_conv_depth), int(w*depths['init']), kernel_size=1),
            'activation': nn.GELU(),
        }),
        'residual1': ConvGroup(int(w*depths['init']),   int(w*depths['block1'])),
        'residual2': ConvGroup(int(w*depths['block1']), int(w*depths['block2'])),
        'residual3': ConvGroup(int(w*depths['block2']), int(w*depths['block3'])),
        'pooling': FastGlobalMaxPooling(),
        'linear': nn.Linear(int(w*depths['block3']), 10, bias=False),
        'temperature': TemperatureScaler(0.1)
    })

    net = SpeedyResNet(network_dict)
    net = net.cuda().to(memory_format=torch.channels_last).train()

    images = next(iter(CifarLoader('cifar10/', train=True, batch_size=5000,
                                   aug=dict(pixelate=pixelate_option))))[0]
    init_whitening_conv(net.net_dict['initial_block']['whiten'], images, pad=2)

    return net

## VGG
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, mult=1):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        self.features = self._make_layers(cfg[vgg_name])
        # this must be changed depending on pixelation option, because VGG flattens after final conv
        self.classifier = nn.Linear(512*mult**2, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def construct_vgg11(pixelate_option):
    a, b = pixelate_option
    assert 32 % a == 0
    mult = (32//a)*b
    return VGG('VGG11', mult)


## ResNet18
from torch import nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)
#         self.linear = nn.Linear(4*512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = 0.1 * out
        out = self.linear(out)
        return out

def construct_rn18(pixelate_option):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=10)

