# -*- coding: utf-8 -*-
"""
    Created on Thursday, Jul 16 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Tuesday, Jul 28 2020

Southeast University, College of Automation, 211189 Nanjing China
"""

import os
import torch
from torch.utils.data import DataLoader
from src.models import SurfaceNet2d
from src.datasets import FakeDataset
from src.utils import CBCELoss, cal_alpha

default_weight_path = '../weights'


def train(epochs, batch_size, check_point, weight_file=''):
    # prepare
    # seed = 1
    # torch.manual_seed(seed)
    os.makedirs(default_weight_path, exist_ok=True)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # initialization
    dataloader = DataLoader(
        FakeDataset(dim=2, length=10),
        batch_size=batch_size,
        shuffle=True
    )
    model = SurfaceNet2d().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00025, momentum=0.9, nesterov=True)
    loss_func = CBCELoss(cal_alpha(dataloader))

    # load weight
    if weight_file == 'latest':
        for root, _, files in os.walk(default_weight_path):
            model.load_state_dict(torch.load(os.path.join(root, files[-1])))
    elif weight_file != '':
        model.load_state_dict(torch.load(weight_file))

    # begin training
    for epoch in range(epochs):
        tt_loss = 0
        for i, (cvc1, cvc2, gt) in enumerate(dataloader):
            cvc1 = cvc1.to(device)
            cvc2 = cvc2.to(device)
            gt = gt.to(device)
            optimizer.zero_grad()
            # forward
            output = model(cvc1, cvc2)
            # compute loss
            loss = loss_func(output, gt)
            tt_loss += loss.item()
            # backward
            loss.backward()
            optimizer.step()

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, tt_loss / len(dataloader.dataset)))
        if (epoch + 1) % check_point == 0:
            torch.save(model.state_dict(), '{}/{:0>4d}.pkl'.format(default_weight_path, epoch + 1))


if __name__ == '__main__':
    train(epochs=10, batch_size=2, check_point=2, weight_file='')



