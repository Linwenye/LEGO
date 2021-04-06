'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

# from utils import progress_bar
# from resnet import *
from liu_models import *
import wandb

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Model
print('==> Building model..')
net = ResNet101_cifar100()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

wandb.init(project="lego")
wandb.watch(net, log="all")

# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

# change checkpoint here
checkpoint = torch.load('./checkpoint/cifar100_res101.pth')
net.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer'])
if args.lr == 0.1:
    scheduler.load_state_dict(checkpoint['scheduler'])
    print('resume lr')
else:
    print('学习率', args.lr)
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

with open('inspect.txt','w') as f:
    for key, value in checkpoint['net'].items():
        f.write("key,", key)
        f.write("value,", value)


