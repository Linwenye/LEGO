.3
'''ReuseNet in PyTorch.

For Pre-activation ReuseNet, see 'preact_ReuseNet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 定义两层
class ReuseBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv2, bn2, stride=1):
        super(ReuseBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv2
        self.bn2 = bn2
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 定义三层
class ReuseBlock3(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ReuseBlock3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ReuseNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=config.num_classes, widen_factor=config.widen_factor):
        super(ReuseNet, self).__init__()
        self.in_planes = 16 * widen_factor

        self.conv1 = nn.Conv2d(3, 16 * widen_factor, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * widen_factor)

        self.layer1 = self._make_layer(block, 16 * widen_factor, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32 * widen_factor, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * widen_factor, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128 * widen_factor, num_blocks[3], stride=2)
        self.linear = nn.Linear(128 * widen_factor * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [1] * (num_blocks - 1)
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion

        reuse_conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        reuse_bn2 = nn.BatchNorm2d(planes)

        for _ in strides:
            layers.append(ReuseBlock2(self.in_planes, planes, reuse_conv2, reuse_bn2))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class CifarReuseNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, widen_factor=config.widen_factor):
        super(CifarReuseNet, self).__init__()
        self.in_planes = 16 * widen_factor

        self.conv1 = nn.Conv2d(3, 16 * widen_factor, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * widen_factor)
        self.layer1 = self._make_layer(block, 16 * widen_factor, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [1] * (num_blocks - 2)
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        layers.append(block(self.in_planes, planes, 1))

        if config.block_layers == 2:
            reuse = BasicBlock(self.in_planes, planes, 1)
        elif config.block_layers == 3:
            reuse = ReuseBlock3(self.in_planes, planes, 1)
        else:
            reuse = None
        for _ in strides:
            layers.append(reuse)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ReuseNet18():
    return ReuseNet(BasicBlock, [2, 2, 2, 2])


def CifarReuseNet26():
    return CifarReuseNet(BasicBlock, [4, 4, 4])


def ReuseNet26():
    return ReuseNet(BasicBlock, [3, 3, 3, 3], num_classes=config.num_classes)


def ReuseNet34():
    return ReuseNet(BasicBlock, [3, 4, 6, 3])


def ReuseNet50():
    return ReuseNet(Bottleneck, [3, 4, 6, 3])


def ReuseNet101():
    return ReuseNet(Bottleneck, [3, 4, 23, 3])


def ReuseNet152():
    return ReuseNet(Bottleneck, [3, 8, 36, 3])


def test():
    # net = ReuseNet34wider_cifar100()
    net = ReuseNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
