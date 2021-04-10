from liu_models import *

lr=0.1
weight_decay=5e-4
widen_factor=4
block_layers=2
train_epoch=220

num_classes=100
net_name='Cifar{}Res32-factored2'.format(num_classes)
net = CifarResNet32(num_classes)
constant_residual_scale = False