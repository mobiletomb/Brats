import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import csv
import glob
from SSL import modellib


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


class LinearKernelCKA():
    def __init__(self, x, y):
        self.dim = x.shape[-1]
        self.x = x
        self.y = y

    def _get_HSIC(self, x, y):
        x = self._build_Linear_Kernel(x)
        y = self._build_Linear_Kernel(y)
        cm = self._build_CM(self.dim)
        tr = torch.trace(cm*x*cm*cm*y*cm)
        return torch.mul(tr, 1 / torch.square((self.dim - 1)))

    def _build_CM(self, dim):
        i = torch.eye(dim)
        ones = torch.ones((dim, dim))
        return i - i * torch.mul(ones, 1./dim)

    def _get_CKA(self):
        hsic_xy = self._get_HSIC(self.x, self.y)
        hsic_xx = self._get_HSIC(self.x, self.x)
        hsic_yy = self._get_HSIC(self.y, self.y)
        return hsic_xy / torch.sqrt(hsic_xx * hsic_yy)

    def _build_Linear_Kernel(self,x):
        K = torch.matmul(x.T, x)
        return K


if __name__ == '__main__':
    inps = torch.arange(5 * 4 * 240 * 240 * 155, dtype=torch.float32).view(5, 4, 240, 240, 155)
    tgts = torch.arange(5 * 4 * 240 * 240 * 155, dtype=torch.float32).view(5, 4, 240, 240, 155)
    dataset = TensorDataset(inps, tgts)
    loader = DataLoader(dataset,
                        batch_size=1,
                        pin_memory=True,
                        num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = modellib.AttGateQuery(4)
    model.to(device)
    for data, target in loader:
        data = data.to(device)
        output = model(data)

    net = nn.Linear(2, 2)
    # 权重矩阵初始化为1
    nn.init.constant_(net.weight, val=100)
    nn.init.constant_(net.bias, val=20)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    print(optimizer.state_dict()['param_groups'])
    for i in range(3):
        loss = []
        for j in range(3):
            loss.append(j)
        print(loss)






