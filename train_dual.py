'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import argparse

# from utils import progress_bar
import wandb
import config_dual

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=config_dual.lr, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if config_dual.num_classes == 10:
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
else:
    trainset = torchvision.datasets.CIFAR100(
        root='./cifar100', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128*2, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./cifar100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100*2, shuffle=False, num_workers=2)
# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# Model
net_name1 = config_dual.net_name1
print('=============> Building model..' + net_name1)

net1 = config_dual.net1
net1 = net1.to(device)
net2 = config_dual.net2
net2 = net2.to(device)
if device == 'cuda':
    net1 = torch.nn.DataParallel(net1)
    net2 = torch.nn.DataParallel(net2)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=config_dual.weight_decay)
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=200)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=config_dual.weight_decay)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=200)

wandb.init(project="lego")


# wandb.watch(net1, log="all")
# wandb.watch(net2, log="all")

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#
#     # change checkpoint here
#     checkpoint = torch.load('./checkpoint/{}.pth'.format(net_name))
#     net.load_state_dict(checkpoint['net'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     if args.lr == config_dual.lr:
#         scheduler.load_state_dict(checkpoint['scheduler'])
#         print('resume lr')
#     else:
#         print('学习率', args.lr)
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
#     print('best_acc,', best_acc)
#     print('start_epoch,', checkpoint['epoch'])
# for key, value in checkpoint['net'].items():
#     print("key,", key)
#     print("value,", value)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net1.train()
    net2.train()
    train_loss = 0
    train_loss2 = 0
    correct = 0
    total = 0
    correct2 = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        outputs = net1(inputs)
        outputs2 = net2(inputs)
        loss = criterion(outputs, targets)
        loss2 = criterion(outputs2, targets)
        loss.backward()
        loss2.backward()
        optimizer1.step()
        optimizer2.step()
        train_loss += loss.item()
        train_loss2 += loss2.item()
        _, predicted = outputs.max(1)
        _, predicted2 = outputs2.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        correct2 += predicted2.eq(targets).sum().item()
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    wandb.log({'train_acc1': 100. * correct / total, 'train_loss1': train_loss, 'train_acc2': 100. * correct2 / total,
               'train_loss2': train_loss2})
    print('train_acc1', 100. * correct / total)
    print('train_loss1', train_loss)
    print('train_acc2', 100. * correct2 / total)
    print('train_loss2', train_loss2)


def test(epoch):
    global best_acc
    net1.eval()
    net2.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_loss2 = 0
    correct2 = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net1(inputs)
            outputs2 = net2(inputs)
            loss = criterion(outputs, targets)
            loss2 = criterion(outputs2, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            test_loss2 += loss2.item()
            _, predicted2 = outputs2.max(1)
            correct2 += predicted2.eq(targets).sum().item()
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    wandb.log({'test_acc1': 100. * correct / total, 'test_loss1': test_loss,'test_acc2': 100. * correct2 / total, 'test_loss2': test_loss2})
    print('test_acc1', 100. * correct / total)
    print('test_loss1', test_loss)

    print('test_acc2', 100. * correct2 / total)
    print('test_loss2', test_loss2)
    # # Save checkpoint.
    # acc = 100. * correct / total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #         'optimizer': optimizer.state_dict(),
    #         'scheduler': scheduler.state_dict()
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/{}.pth'.format(net_name))
    #     best_acc = acc


#
# if args.resume:
#     test(start_epoch)
def swap_layer(bernu):
    if bernu[0]:
        net1.layer1, net2.layer1 = net2.layer1, net1.layer1
    if bernu[1]:
        net1.layer2, net2.layer2 = net2.layer2, net1.layer2
    if bernu[2]:
        net1.layer3, net2.layer3 = net2.layer3, net1.layer3
    if bernu[3]:
        net1.layer4, net2.layer4 = net2.layer4, net1.layer4


for epoch in range(start_epoch, start_epoch + config_dual.train_epoch):
    '''
        random and swap
        temp: swap layer
        further: swap block
        further er: block co-supervision
        other: deep self supervision
    '''
    swap_prob = torch.Tensor([0.25] * 4)
    bernu = torch.bernoulli(swap_prob)
    print('swap layer tensor', bernu)

    swap_layer(bernu)
    train(epoch)
    swap_layer(bernu)

    test(epoch)

    scheduler1.step()
    scheduler2.step()
    print('lr:', scheduler1.get_last_lr())
