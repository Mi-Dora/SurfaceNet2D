# -*- coding: utf-8 -*-
"""
    Created on Thursday, Jul 16 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Tuesday, Jul 28 2020

Southeast University, College of Automation, 211189 Nanjing China
"""

from torch.utils import data
import torch


class FakeDataset(data.Dataset):
    """
    Load custom dataset by giving the data folder path
    """

    def __init__(self, dim, length):
        """
        :param dim: (int) The dimension of the fake data, expected to be 1 or 2
        :param length: (int) length of the fake dataset
        """
        super(FakeDataset, self).__init__()
        assert dim == 2 or dim == 3
        self.dimension = dim
        self.length = length
        self.cvc1s = []
        self.cvc2s = []
        self.gts = []
        for _ in range(length):
            self.gen_rand_data()

    def __getitem__(self, idx):
        """
        return: (tensor) data send to network
        obj[idx] == obj.__getitem__(idx)
        """
        return self.cvc1s[idx], self.cvc2s[idx], self.gts[idx]

    def __len__(self):
        return self.length

    def gen_rand_data(self):
        if self.dimension == 2:
            self.cvc1s.append(torch.rand(3, 100, 100))
            self.cvc2s.append(torch.rand(3, 100, 100))
            self.gts.append(torch.randint(0, 2, (1, 100, 100), dtype=torch.float))
        else:
            self.cvc1s.append(torch.rand(3, 100, 100, 100))
            self.cvc2s.append(torch.rand(3, 100, 100, 100))
            self.gts.append(torch.randint(0, 2, (1, 100, 100, 100), dtype=torch.float))


if __name__ == '__main__':

    dataloader = data.DataLoader(
        FakeDataset(1, 10),
        batch_size=2,
        shuffle=True
    )
    for i, (cvc1, cvc2, gt) in enumerate(dataloader):
        print(i)
        print(cvc1.shape)
        print(cvc2.shape)
        print(gt.shape)
