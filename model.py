import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from os.path import join
import os
import torch

class DeepBALM(nn.Module):
    def __init__(self, num_layer):
        super(DeepBALM, self).__init__()
        self.num_layer = num_layer
        onelayer = []

        for i in range(self.num_layer):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
    
    def forward(self, x, y, A, b):
        for i in range(self.num_layer):
            x, y = self.fcs[i](x, y, A, b)

        return x, y


class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.r = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.H_inv = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(1089, 1089)))
        
        self.conv1_forward = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2_forward = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        
    
        
    def forward(self, x_k, y_k, A, b):

        q0k = x_k + 1 / self.r * torch.matmul(A.T, y_k)
        q0k_input = q0k.view(-1, 1, 33, 33)
        x_k1 = F.conv2d(q0k_input, self.conv1_forward, padding=1)
        x_k1 = F.relu(x_k1)
        x_k1 = F.conv2d(x_k1, self.conv2_forward, padding=1)

        x_k1 = torch.mul(torch.sign(x_k1), F.relu(torch.abs(x_k1) - self.soft_thr))

        x_k1 = F.conv2d(x_k1, self.conv1_backward, padding=1)
        x_k1 = F.relu(x_k1)
        x_k1 = F.conv2d(x_k1, self.conv2_backward, padding=1)

        x_k1 = x_k1.view(-1, 1089, 1)
        s0k = torch.matmul(A, (2*x_k1 - x_k) - b)

        y_k1 = y_k - torch.matmul(self.H_inv, s0k)

        return x_k1, y_k1




model = DeepBALM(5)
summary(model, [(10,1089,1), (1089,1), (1089, 1089), (1089, 1)])
