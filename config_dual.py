from liu_models import *

lr=0.1
weight_decay=5e-4
widen_factor=4
block_layers=2
train_epoch=220

num_classes=100
net_name1 = 'ResNet18-1'
net_name2 = 'ResNet18-2'
net1 = ResNet18(num_classes)
net2 = ResNet18(num_classes)
constant_residual_scale = True