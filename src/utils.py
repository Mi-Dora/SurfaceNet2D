# -*- coding: utf-8 -*-
"""
    Created on Thursday, Jul 16 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Tuesday, Jul 28 2020

Southeast University, College of Automation, 211189 Nanjing China
"""

import torch
import torch.nn as nn

NEAR_0 = 1e-10


def cal_alpha(dataloader):
    """
    Calculate class-balance coefficient alpha
    """
    alpha = 0
    for i, (_, _, gt) in enumerate(dataloader):
        voxel_num = 1
        for s in gt.shape:
            voxel_num *= s
        alpha += 1 - gt.sum() / voxel_num
    alpha /= len(dataloader)
    print(alpha)
    return alpha


class CBCELoss(nn.Module):
    """
    Class-balanced cross-entropy loss function
    """
    def __init__(self, alpha):
        super(CBCELoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, gt):
        loss1 = -self.alpha * gt * torch.log(output + NEAR_0)
        loss2 = -(1-self.alpha) * (1 - gt) * torch.log(1 - output + NEAR_0)
        loss = (loss1 + loss2).sum()
        return loss


if __name__ == '__main__':
    # random input data
    output = torch.rand(8, 1, 100, 100, requires_grad=True)
    gt = torch.tensor(torch.randint(0, 2, (8, 1, 100, 100)), dtype=torch.float, requires_grad=True)
    loss_func = CBCELoss(0.4)
    loss = loss_func(output, gt)
    print(loss)

