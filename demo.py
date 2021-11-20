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
    # inps = torch.arange(5 * 4 * 240 * 240 * 155, dtype=torch.float32).view(5, 4, 240, 240, 155)
    # tgts = torch.arange(5 * 4 * 240 * 240 * 155, dtype=torch.float32).view(5, 4, 240, 240, 155)
    # dataset = TensorDataset(inps, tgts)
    # loader = DataLoader(dataset,
    #                     batch_size=1,
    #                     pin_memory=True,
    #                     num_workers=2)
    #
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = modellib.AttGateQuery(4)
    # model.to(device)
    # for data, target in loader:
    #     data = data.to(device)
    #     output = model(data)
    #
    # net = nn.Linear(2, 2)
    # # 权重矩阵初始化为1
    # nn.init.constant_(net.weight, val=100)
    # nn.init.constant_(net.bias, val=20)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    # print(optimizer.state_dict()['param_groups'])
    for i in range(3):
        loss = []
        for j in range(3):
            loss.append(j)
        print(loss)
    # # query = torch.autograd.Variable(torch.ones(1, 5), requires_grad=True).to(device)
    # for name, para in model.named_parameters():
    #     if name == 'fc1.weight':
    #         para.requires_grad = False
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.9)
    #
    #
    # for i in range(2):
    #     for batch_ndx, sample in enumerate(loader):
    #         for name, para in model.named_parameters():
    #             if name == 'fc1.weight':
    #                 print('fc1:{}'.format(para))
    #             # if name == 'fc2.weight':
    #             #     print('fc2:{}'.format(para))
    #         # print(query)
    #         optimizer.zero_grad()
    #         inps = sample[0].to(device)
    #         targets = sample[1].to(device)
    #         log = model(inps)
    #         # log = torch.multiply(log, query)
    #         loss = F.binary_cross_entropy_with_logits(log, targets)
    #         loss.backward()
    #         optimizer.step()

    # 全局池化测试
    # pool of square window of size=3, stride=2
    # input = torch.randn(20, 16, 50, 44, 31)
    # m = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
    # output = m(input)
    # print(output.shape)


    # loader, optimizer, model, loss_fn = ...
    # swa_model = torch.optim.swa_utils.AveragedModel(model)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    # swa_start = 160
    # swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    #
    # for epoch in range(300):
    #        for input, target in loader:
    #           optimizer.zero_grad()
    #            loss_fn(model(input), target).backward()
    #            optimizer.step()
    #       if epoch > swa_start:
    #            swa_model.update_parameters(model)
    #            swa_scheduler.step()
    #        else:
    #            scheduler.step()
    #
    # # Update bn statistics for the swa_model at the end
    # torch.optim.swa_utils.update_bn(loader, swa_model)
    # # Use swa_model to make predictions on test data
    # preds = swa_model(test_input)