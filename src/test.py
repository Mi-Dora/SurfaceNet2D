# -*- coding: utf-8 -*-
"""
    Created on Thursday, Jul 16 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Wednesday, Jul 29 2020

Southeast University, College of Automation, 211189 Nanjing China
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from src.models import SurfaceNet2d
from src.datasets import FakeDataset


default_weight_path = '../weights'
default_image_path = '../images'


def test(batch_size=1, weight_file=''):
    # prepare
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # initialization
    dataloader = DataLoader(
        FakeDataset(dim=2, length=10),
        batch_size=batch_size,
        shuffle=True
    )
    model = SurfaceNet2d().to(device)

    # load weight
    if weight_file == 'latest':
        for root, _, files in os.walk(default_weight_path):
            model.load_state_dict(torch.load(os.path.join(root, files[-1])))
    elif weight_file != '':
        model.load_state_dict(torch.load(weight_file))

    # begin training
    for i, (cvc1, cvc2, _) in enumerate(dataloader):
        cvc1 = cvc1.to(device)
        cvc2 = cvc2.to(device)
        # forward
        output = model(cvc1, cvc2)
        save_image(output.data, "{0}/{1}.png".format(default_image_path, i), nrow=1, normalize=True)
        print("{0}/{1}.png saved.".format(default_image_path, i))


if __name__ == '__main__':
    os.makedirs(default_image_path, exist_ok=True)
    test(batch_size=1, weight_file='')


