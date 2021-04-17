import torch

swap_prob = torch.Tensor([0.25] * 4)
bernu = torch.bernoulli(swap_prob)
print(bernu)