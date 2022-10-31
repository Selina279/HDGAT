import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


class MLP(Module):
    def __init__(self, in_d, out_d):
        super(MLP, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        hidden = in_d / out_d
        hidden1 = int(in_d / math.sqrt(hidden))
        hidden2 = int(hidden1 / math.sqrt(hidden))
        self.l1 = torch.nn.Linear(in_d, hidden1)
        self.l2 = torch.nn.Linear(hidden1, hidden2)
        self.l3 = torch.nn.Linear(hidden2, out_d)

    def forward(self, inputs):
        out = F.relu(self.l1(inputs))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return F.softmax(out, dim=1)


class SelfAttention(Module):
    """docstring for SelfAttention"""
    def __init__(self, in_features, idx, hidden_dim):
        super(SelfAttention, self).__init__()
        self.idx = idx
        self.linear = torch.nn.Linear(in_features, hidden_dim)
        self.a = Parameter(torch.FloatTensor(2 * hidden_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.a.size(1))
        self.a.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        # input size: node_num * 3 * in_features
        x = self.linear(inputs).transpose(0, 1)
        self.n = x.size()[0]
        x = torch.cat([x, torch.stack([x[self.idx]] * self.n, dim=0)], dim=2)
        U = torch.matmul(x, self.a).transpose(0, 1)
        U = F.leaky_relu_(U)  # 非线性激活
        weights = F.softmax(U, dim=1)
        outputs = torch.matmul(weights.transpose(1, 2), inputs).squeeze(1) * 3
        return outputs, weights