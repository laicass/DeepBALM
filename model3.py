import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from os.path import join
import os
import torch

class DeepBALM(nn.Module):
    def __init__(self, m, n, num_layer):
        super(DeepBALM, self).__init__()
        self.num_layer = num_layer
        self.m = m
        self.n = n
        onelayer = []

        for i in range(self.num_layer):
            onelayer.append(PrimalBlock())
            onelayer.append(ProxBlock(n, num_filter=16))
            onelayer.append(LeastSquareBlock())
            onelayer.append(DualBlock())

        self.fcs = nn.ModuleList(onelayer)
    
    def forward(self, x, y, z, A, b):
        x_list = [x,]
        for i in range(0, 4 * self.num_layer, 4):
            q0k = self.fcs[i](x, z, A, b)
            x_k1 = self.fcs[i+1](q0k)
            y_k1 = self.fcs[i+2](x_k1, y, A, b)
            z_k1 = self.fcs[i+3](y, y_k1, z, A, b)
            x = x_k1
            y = y_k1
            z = z_k1
            x_list.append(x)
        final_x = x
        final_z = z_k1
        return final_x, final_z, x_list

class PrimalBlock(nn.Module):
    def __init__(self):
        super(PrimalBlock, self).__init__()
        self.param1 = nn.Parameter(torch.Tensor([0.01]))

    def forward(self, x, z, A, b):
        q0k = x + self.param1 * torch.matmul(A.T, z)
        return q0k


class ProxBlock(nn.Module):
    def __init__(self, n, num_filter=32, size=3):
        super(ProxBlock, self).__init__()
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.conv1_forward = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(num_filter, 1, size, size)))
        self.conv2_forward = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(num_filter, num_filter, size, size)))
        self.conv1_backward = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(num_filter, num_filter, size, size)))
        self.conv2_backward = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(1, num_filter, size, size)))
        self.n = n
        self.root_n = int(n**0.5)
        
        
    def forward(self, q0k):
        q0k_input = q0k.view(-1, 1, self.root_n, self.root_n)
        x_k1 = F.conv2d(q0k_input, self.conv1_forward, padding=1)
        x_k1 = F.relu(x_k1)
        x_k1_forward = F.conv2d(x_k1, self.conv2_forward, padding=1)

        x_k1 = torch.mul(torch.sign(x_k1_forward), F.relu(torch.abs(x_k1_forward) - self.soft_thr))

        x_k1 = F.conv2d(x_k1, self.conv1_backward, padding=1)
        x_k1 = F.relu(x_k1)
        x_k1_backward = F.conv2d(x_k1, self.conv2_backward, padding=1)

        x_k1_pred = x_k1_backward.view(-1, self.n, 1)
        return x_k1_pred

class LeastSquareBlock(nn.Module):
    def __init__(self):
        super(LeastSquareBlock, self).__init__()
        self.param2 = nn.Parameter(torch.Tensor([0.01]))
    
    def forward(self, x_k1, y, A, b):   
        y_k1 = y - self.param2 * torch.matmul(A.T, (torch.matmul(A, x_k1) - b))
        return y_k1
    

class DualBlock(nn.Module):
    def __init__(self):
        super(DualBlock, self).__init__()
        self.param3 = nn.Parameter(torch.Tensor([0.01]))
    
    def forward(self, y, y_k1, z, A, b):
        s0k = torch.matmul(A, 2*y_k1 - y) - b
        #y_k1 = y - torch.matmul(self.H_inv, s0k)        
        z_k1 = z - self.param3 * s0k
        return z_k1


